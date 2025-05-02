# This script will collect responses from Qwen Assistant for labeling purposes.abs
import datasets
import openai
from copy import deepcopy
from dev.induced_instruction.tools import OPENAI_SCHEMAS, TOOL_FUNCTIONS
import json

from tqdm.contrib.concurrent import thread_map

QWEN_SCHEMA = list(OPENAI_SCHEMAS.values())

dataset = datasets.load_from_disk("dataset_v2/user_queries").shuffle(seed=52).to_list()[:40]

client = openai.OpenAI(api_key="None", base_url="http://localhost:8000/v1")
model_name = client.models.list().data[0].id


def tool_use_loop(messages, new_instruction):
    messages = deepcopy(messages)
    while True:
        response = client.chat.completions.create(
            model=model_name,
            messages=messages,
            tools=QWEN_SCHEMA,
            temperature=0.00,
            max_tokens=2048,
        )

        choice = response.choices[0]
        if choice.message.content is None:
            choice.message.content = ""

        messages.append(choice.message.model_dump(mode="json"))
        if choice.finish_reason == "tool_calls":
            passed = 0
            for tool_call in choice.message.tool_calls:
                tid = tool_call.id
                fn_name = tool_call.function.name
                fn_args = json.loads(tool_call.function.arguments)

                if fn_name in TOOL_FUNCTIONS:
                    response = TOOL_FUNCTIONS[fn_name](**fn_args)
                    qwen_response = json.dumps(response)

                    messages.append(
                        {
                            "role": "tool",
                            "content": qwen_response,
                            "tool_call_id": tid,
                        }
                    )
                    passed += 1
            if passed == 0:
                with open("tmp.json", "a") as f:
                    f.write(json.dumps({"messages": messages, "new_instruction": new_instruction}) + "\n")
                return messages
        else:
            with open("tmp.json", "a") as f:
                f.write(json.dumps({"messages": messages, "new_instruction": new_instruction}) + "\n")
            return messages


def get_qwen_response(data_row):
    messages = data_row["messages"]
    messages = tool_use_loop(messages, data_row["new_instruction"])

    return messages


with open("tmp.json", "w") as f:
    f.write("")
    pass

thread_map(
    get_qwen_response,
    dataset,
    max_workers=10,
    desc="Collecting Qwen Assistant responses",
    total=len(dataset),
)

def reformat_message(message):
    if "tool_calls" not in message:
        return {
            "role": message["role"],
            "content": [
                {"type": "text", "text": message["content"]},
            ],
        }
    else:
        return {
            "role": message["role"],
            "content": [
                {"type": "text", "text": message["content"]},
            ]
            + [
                {
                    "type": "tool_use",
                    "id": tool_call["id"],
                    "name": tool_call["function"]["name"],
                    "input": json.loads(tool_call["function"]["arguments"]),
                }
                for tool_call in message["tool_calls"]
            ],
        }

def reformat_messages(messages):
    return [
        reformat_message(message)
        for message in messages
    ]

# save as an actual dataset
with open("tmp.json", "r") as f:
    lines = f.readlines()
    dataset = [json.loads(line) for line in lines]

# convert the dataset to similar to the claude messages format
result = []
for row in dataset:
    result.append({
        "messages": reformat_messages(row["messages"]),
        "new_instruction": row["new_instruction"]
    })

ds = datasets.Dataset.from_list(result)
print(ds)
ds.save_to_disk("dataset_v2/qwen_assistant_responses")