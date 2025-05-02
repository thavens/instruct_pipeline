# pydantic messages class
from __future__ import annotations
from typing import Optional
from pydantic import BaseModel

import datasets
from rich.console import Console
from rich.prompt import Confirm
from rich.panel import Panel

import os
import json


class Message(BaseModel):
    role: str
    content: str


class Data(BaseModel):
    messages: list[Message]
    new_instruction: str
    instruction_judgment: Optional[bool] = None
    user_query_judgment: Optional[bool] = None
    assistant_response_judgment: Optional[bool] = None

def flatten_content_block(anthropic_message: dict) -> str:
    content = ""
    for block in anthropic_message["content"]:
        if block["type"] == "text":
            content += block["text"]
    return content

def flatten_content_blocks(anthropic_messages: list[dict]) -> list[Message]:
    system_msg = Message(
        role=anthropic_messages[0]["role"],
        content=flatten_content_block(anthropic_messages[0]),
    )
    user_msg = Message(
        role=anthropic_messages[1]["role"],
        content=flatten_content_block(anthropic_messages[1]),
    )
    content = ""
    for anthropic_message in anthropic_messages[2:]:
        if anthropic_message["role"] == "assistant":
            content += flatten_content_block(anthropic_message)
    assist_msg = Message(role=anthropic_message["role"], content=content)
    return [system_msg, user_msg, assist_msg]


dataset = datasets.load_from_disk("dataset_v2/qwen_assistant_responses")
dataset = dataset.to_list()

formatted_data = []
for data_row in dataset:
    flat_messages = flatten_content_blocks(data_row["messages"])
    row = Data(messages=flat_messages, new_instruction=data_row["new_instruction"])
    formatted_data.append(row)

# grade the dataset rows with an interactive UI, with resume and checkpoint support
console = Console()


# Path to checkpoint directory
CHECKPOINT_PATH = "dataset_v2/qwen_judged_responses.jsonl"

# Load existing progress if available
results: list[Data] = []
if os.path.exists(CHECKPOINT_PATH):
    with open(CHECKPOINT_PATH, "r") as f:
        existing_list = [json.loads(line) for line in f.readlines()]
    for item in existing_list:
        results.append(Data.model_validate(item))
    console.print(f"Resuming from previous checkpoint, {len(results)} entries loaded.")
else:
    console.print("No checkpoint found. Starting fresh.")

for idx, row in enumerate(formatted_data, start=1):
    # Skip already processed entries
    if idx <= len(results):
        continue
    # Divider for each entry
    console.rule(f"[bold green]Row {idx}")
    # Judge the new instruction
    console.print(
        Panel(row.new_instruction, title="New Instruction", style="bold magenta")
    )
    row.instruction_judgment = Confirm.ask("Is the instruction verifiable?")
    # Judge the user query
    user_query = row.messages[1].content if len(row.messages) > 1 else ""
    console.print(Panel(user_query, title="User Query", style="bold yellow"))
    row.user_query_judgment = Confirm.ask("Is the user query verifiable?")
    # Judge the assistant response against the instruction
    console.rule("[bold green]Instruction vs Assistant Response")
    console.print(
        Panel(row.new_instruction, title="New Instruction", style="bold magenta")
    )
    assistant_resp = row.messages[2].content if len(row.messages) > 2 else ""
    console.print(Panel(assistant_resp, title="Assistant Response", style="bold cyan"))
    row.assistant_response_judgment = Confirm.ask(
        "Is the assistant response following the instruction?"
    )
    console.print()  # blank line

    # Append result and save progress
    results.append(row)
    with open(CHECKPOINT_PATH, "a") as f:
        f.write(row.model_dump_json() + "\n")

# Final save after processing all entries
console.print("All entries processed. Saving final dataset...")
final_ds = datasets.Dataset.from_list([f.model_dump(mode="json") for f in results])
data_path = "dataset_v2/qwen_judged_responses"
final_ds.save_to_disk(data_path)
console.print(f"Final dataset saved to {data_path}.")
