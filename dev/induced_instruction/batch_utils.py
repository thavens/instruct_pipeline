import time
from copy import deepcopy
import json

import anthropic
import dotenv
from anthropic.types import Message, ToolUseBlock
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatch
from anthropic.types.messages.batch_create_params import Request
from rich.progress import Progress
from tenacity import Retrying, stop_after_attempt, wait_fixed

from dev.induced_instruction.tools import CLAUDE_SCHEMAS, OPENAI_SCHEMAS, TOOL_FUNCTIONS

dotenv.load_dotenv()

client = anthropic.Anthropic()


def completion(msgs: list[dict]) -> str:
    if msgs[0]["role"] == "system":
        response = (
            anthropic.Anthropic()
            .messages.create(
                model="claude-3-5-haiku-20241022",
                temperature=0.1,
                top_p=0.8,
                max_tokens=1024,
                system=msgs[0]["content"],
                messages=msgs[1:],
            )
            .content[0]
            .text
        )
    else:
        response = (
            anthropic.Anthropic()
            .messages.create(
                model="claude-3-5-haiku-20241022",
                temperature=0.1,
                top_p=0.8,
                max_tokens=1024,
                messages=msgs,
            )
            .content[0]
            .text
        )
    return response


def batch_request(msgs_batch: list[dict], tools=False) -> str:
    requests = []
    for idx, messages in enumerate(msgs_batch):
        if messages[0]["role"] == "system":
            req = Request(
                custom_id=f"{idx}",
                params=MessageCreateParamsNonStreaming(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1024,
                    temperature=0.1,
                    top_p=0.8,
                    system=messages[0]["content"],
                    messages=messages[1:],
                    tools=CLAUDE_SCHEMAS if tools else [],
                ),
            )
        else:
            req = Request(
                custom_id=f"{idx}",
                params=MessageCreateParamsNonStreaming(
                    model="claude-3-5-haiku-20241022",
                    max_tokens=1024,
                    temperature=0.1,
                    top_p=0.8,
                    messages=messages,
                    tools=CLAUDE_SCHEMAS if tools else [],
                ),
            )
        requests.append(req)

    status = client.messages.batches.create(requests=requests)
    return status


def wait_for_batch(batch_id: str, progress: Progress) -> list[Message]:
    retryer = Retrying(stop=stop_after_attempt(5), wait=wait_fixed(5), reraise=True)
    status: MessageBatch = retryer(client.messages.batches.retrieve, batch_id)

    total = sum(status.request_counts.to_dict().values())
    task_id = progress.add_task(batch_id, total=total)
    while status.processing_status == "in_progress":
        status: MessageBatch = retryer(client.messages.batches.retrieve, status.id)
        progress.update(task_id, completed=status.request_counts.succeeded)
        time.sleep(1)

    status = client.messages.batches.retrieve(status.id)
    results = client.messages.batches.results(status.id)

    # sort the results on the custom_id
    results = list(results)
    results.sort(key=lambda x: int(x.custom_id))

    responses = []
    for idx, result in enumerate(results):
        responses.append(
            result.result.message if result.result.type == "succeeded" else None
        )

    return responses


def batch_complete(msgs_batch: list[dict], progress: Progress) -> list[str]:
    msgs_batch = deepcopy(msgs_batch)
    repeat = True
    active = list(range(len(msgs_batch)))
    responses = ["" for i in range(len(msgs_batch))]

    while repeat:
        repeat = False
        status = batch_request(
            [msgs_batch[idx] for idx in range(len(msgs_batch)) if idx in active],
            tools=False,
        )
        msgs_batch_response: list[Message] = wait_for_batch(status.id, progress)

        active_new = []  # activate is map where index is the generation and the value is the corresponding msgs in msgs_batch
        for idx, msg in enumerate(msgs_batch_response):
            messages_idx = active[idx]
            if msg is None:  # failed to generate
                repeat = True
                active_new.append(messages_idx)
            else:
                response_str = ""
                for block in msg.content:
                    if block.type == "text":
                        response_str += block.text
                    else:
                        raise NotImplementedError(
                            f"block type of {block.type} is not handled"
                        )
                responses[messages_idx] = response_str
        active = active_new
    return responses


def batch_messages_complete(
    msgs_batch: list[dict], progress: Progress, tools=False, log_output=None
) -> list[dict]:
    msgs_batch = deepcopy(msgs_batch)
    active = list(range(len(msgs_batch)))
    
    i = 0
    while len(active) > 0:
        i += 1
        if i > 10:
            raise RuntimeError("Too many iterations in batch_messages_complete")
        status = batch_request(
            [msgs_batch[idx] for idx in range(len(msgs_batch)) if idx in active],
            tools=tools,
        )
        if log_output is not None:
            with open(log_output / f"{status.id}.json", "w") as f:
                for idx in active:
                    f.write(json.dumps(msgs_batch[idx]) + "\n")
        msgs_batch_response: list[Message] = wait_for_batch(status.id, progress)

        active_new = []  # active is map where index is the generation and the value is the corresponding msgs in msgs_batch
        for idx, msg in enumerate(msgs_batch_response):
            messages_idx = active[idx]
            if msg is None:  # failed to generate
                active_new.append(messages_idx)
            # following https://docs.anthropic.com/en/docs/build-with-claude/tool-use/overview
            elif msg.stop_reason == "tool_use":
                tool_calls: list[ToolUseBlock] = []
                for block in msg.content:
                    if block.type == "tool_use":
                        tool_calls.append(block)

                # check parsing
                parse_failed = False
                for tool in tool_calls:
                    required_args = OPENAI_SCHEMAS[tool.name]["function"]["parameters"][
                        "required"
                    ]
                    parse_pass = all([i in tool.input for i in required_args])
                    parse_failed = not parse_pass or parse_failed
                    if not parse_pass:
                        print("[WARNING] Parsing failed. Attempting again")

                if not parse_failed:
                    tool_results = []
                    for tool in tool_calls:
                        tool_result = TOOL_FUNCTIONS[tool.name](**tool.input)
                        tool_result["type"] = "tool_result"
                        tool_result["tool_use_id"] = tool.id
                        tool_results.append(tool_result)

                    json_content = [
                        block.model_dump(exclude="citations", mode="json")
                        for block in msg.content
                    ]
                    msgs_batch[messages_idx].append(
                        {"role": msg.role, "content": json_content}
                    )
                    
                    if len(tool_results) > 0:
                        msgs_batch[messages_idx].append(
                            {"role": "user", "content": tool_results}
                        )
                        active_new.append(messages_idx)
                    else:
                        print("[WARNING] No tool results. Returning assistant text only.")
                else:
                    # only keep this conversation active if we have tool results, or parsing failed so we can generate a new one.
                    active_new.append(messages_idx)
            else:
                json_content = [
                    block.model_dump(exclude="citations", mode="json")
                    for block in msg.content
                ]
                msgs_batch[messages_idx].append(
                    {"role": msg.role, "content": json_content}
                )
        active = active_new
    return msgs_batch
