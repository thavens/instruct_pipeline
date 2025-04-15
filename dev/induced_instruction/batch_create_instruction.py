import argparse
import random
import re
import time

import anthropic
import anthropic.types.messages.message_batch
import datasets
import dotenv
import tqdm
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages.batch_create_params import Request

from dev.prompts import (
    GENERATE_USERS_CONFLICTING_PROMPT,
    GET_VERIFIER_CLAUSES_PROMPT,
)

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


def batch_request(msgs_batch: list[dict]) -> str:
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
                ),
            )
        requests.append(req)

    status = client.messages.batches.create(requests=requests)
    return status


def wait_for_batch(msgs_batch: list[dict], batch_id: str):
    status = client.messages.batches.retrieve(batch_id)

    total = sum(status.request_counts.to_dict().values())
    pbar = tqdm.tqdm(total=total, desc="Batch Processing", unit="message")
    while status.processing_status == "in_progress":
        status = client.messages.batches.retrieve(status.id)
        pbar.update(status.request_counts.succeeded)
        pbar.refresh()
        time.sleep(1)
    pbar.close()

    status = client.messages.batches.retrieve(status.id)
    results = client.messages.batches.results(status.id)

    # sort the results on the custom_id
    results = list(results)
    results.sort(key=lambda x: int(x.custom_id))
    passed = sum(1 for result in results if result.result.type == "succeeded")
    print(f"Batch processing completed. {passed}/{total} succeeded.")

    if passed < total:
        pbar = tqdm.tqdm(
            total=total - passed, desc="Processing Results", unit="message"
        )
    i = 0
    responses = []
    for idx, result in enumerate(results):
        if result.result.type == "succeeded":
            response_str = []
            for content in result.result.message.content:
                if content.type == "text":
                    response_str.append(content.text)
            responses.append("".join(response_str))
        else:
            i += 1
            responses.append(completion(msgs_batch[idx]))
            pbar.update(i)
    if passed < total:
        pbar.close()

    return responses


def batch_complete(msgs_batch: list[dict]) -> str:
    status = batch_request(msgs_batch)
    return wait_for_batch(msgs_batch, status.id)


def insert_sentence_at_random_period(base_prompt: str, sentence: str) -> str:
    # add a period to sentence if there is none before
    sentence += "." if sentence[-1] != "." and sentence[-1] != "?" and sentence[-1] != "!" else ""
    sentence = sentence.strip()
    
    # Find all period positions in the base prompt using regex
    period_matches = list(re.finditer(r"\.", base_prompt))

    # If there are no periods, just append the sentence
    if not period_matches:
        return f"{base_prompt}\n{sentence}"

    # Choose a random period match
    chosen_match = random.choice(period_matches)
    start, end = chosen_match.span()

    # Construct the new prompt
    new_prompt = base_prompt[:end] + f" {sentence}" + base_prompt[end:]
    return new_prompt


personas = datasets.load_dataset("proj-persona/PersonaHub", "persona", split="train")[
    "persona"
]


def adv_user_prompt(data_row: dict) -> str:
    sys_prompt = data_row["messages"][0]["content"]
    new_instruction = data_row["new_instruction"]
    prompt = GENERATE_USERS_CONFLICTING_PROMPT.format(
        system=sys_prompt,
        instruction=new_instruction,
        persona=personas[int(random.random() * len(personas))],
    )
    messages = [{"role": "user", "content": prompt}]
    return messages


def emphasize_instruction(data_row: dict) -> list[dict]:
    user_msg = (
        data_row["messages"][1]["content"]
        + f"\nMake sure to keep in mind that the system prompt states \"\"\"{data_row['new_instruction']}\"\"\""
    )
    messages = [
        {"role": "system", "content": data_row["messages"][0]["content"]},
        {"role": "user", "content": user_msg},
    ]
    return messages


# parse arguments for different parts of the pipeline: [get clauses, induce instructions, generate user queries, generate assistant responses]
parser = argparse.ArgumentParser(description="Batch create instructions.")
parser.add_argument(
    "--get_clauses",
    action="store_true",
    help="Get verifiable clauses from the dataset.",
)
parser.add_argument(
    "--generate_user_queries",
    action="store_true",
    help="Generate user queries from the dataset.",
)
parser.add_argument(
    "--generate_assistant_responses",
    action="store_true",
    help="Generate assistant responses from the dataset.",
)

args = parser.parse_args()

if args.get_clauses:
    # get clauses from the dataset
    dataset = datasets.load_dataset("normster/RealGuardrails", "prompts", split="train")
    dataset = dataset.filter(lambda x: x["is_sft"], keep_in_memory=True)
    instructions = dataset["instructions"][:10]

    dataset = []
    for sys_prompt in instructions:
        full_prompt = GET_VERIFIER_CLAUSES_PROMPT.format(system=sys_prompt)
        dataset.append([{"role": "user", "content": full_prompt}])

    responses = batch_complete(dataset)
    clause_data = []
    for sys_prompt, response in zip(instructions, responses):
        matches = re.findall(r"(?s)<clause>(.*?)</clause>", response)
        matches = [match.strip() for match in matches]
        for match in matches:
            new_sys_prompt = insert_sentence_at_random_period(sys_prompt, match)
            clause_data.append(
                {
                    "messages": [
                        {"role": "system", "content": new_sys_prompt},
                    ],
                    "new_instruction": match,
                }
            )
    new_dataset = datasets.Dataset.from_list(clause_data)
    new_dataset.save_to_disk("outputs/clauses")
    print(new_dataset)


if args.generate_user_queries:
    # generate user queries from the dataset
    dataset = datasets.load_from_disk("outputs/clauses")

    msgs_batch = []
    for row in dataset:
        msgs_batch.append(adv_user_prompt(row))

    user_queries = batch_complete(msgs_batch)
    user_query_data = []
    for data_row, response in zip(dataset, user_queries):
        user_query_data.append(
            {
                "messages": [
                    {"role": "system", "content": data_row["messages"][0]["content"]},
                    {"role": "user", "content": response},
                ],
                "new_instruction": data_row["new_instruction"],
            }
        )
    new_dataset = datasets.Dataset.from_list(user_query_data)
    new_dataset.save_to_disk("outputs/user_queries")
    print(new_dataset)


if args.generate_assistant_responses:
    # generate assistant responses from the dataset
    dataset = datasets.load_from_disk("outputs/user_queries")

    msgs_batch = []
    for row in dataset:
        msgs_batch.append(emphasize_instruction(row))

    assistant_responses = batch_complete(msgs_batch)
    assistant_response_data = []
    for data_row, response in zip(dataset, assistant_responses):
        assistant_response_data.append(
            {
                "messages": [
                    {"role": "system", "content": data_row["messages"][0]["content"]},
                    {"role": "user", "content": data_row["messages"][1]["content"]},
                    {"role": "assistant", "content": response},
                ],
                "new_instruction": data_row["new_instruction"],
            }
        )
    datasets.Dataset.from_list(assistant_response_data).save_to_disk(
        "outputs/assistant_responses"
    )
