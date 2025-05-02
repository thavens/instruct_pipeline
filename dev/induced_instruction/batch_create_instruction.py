# This is a script that formulates the complete generation pipeline for generating user assistant
# interactions stress the new instruction and generate an SFT dataset.

import argparse
import os
import random
import re
from pathlib import Path
from itertools import batched

import datasets
from rich.progress import Progress, MofNCompleteColumn, TextColumn
from rich import progress
import dotenv

from dev.induced_instruction.batch_utils import batch_complete, batch_messages_complete
from dev.prompts import (
    GENERATE_USERS_CONFLICTING_PROMPT,
    GET_VERIFIER_CLAUSES_PROMPT,
)

dotenv.load_dotenv()


def insert_sentence_at_random_period(base_prompt: str, sentence: str) -> str:
    # add a period to sentence if there is none before
    sentence += (
        "."
        if sentence[-1] != "." and sentence[-1] != "?" and sentence[-1] != "!"
        else ""
    )
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


def adv_user_prompt(data_row: dict) -> list[dict]:
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
        {
            "role": "system",
            "content": [{"type": "text", "text": data_row["messages"][0]["content"]}],
        },
        {"role": "user", "content": [{"type": "text", "text": user_msg}]},
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
parser.add_argument(
    "--output_dir",
    type=Path,
    default="outputs",
    help="Directory to save the output dataset.",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=100,
    help="Batch size for processing.",
)

args = parser.parse_args()
os.makedirs(args.output_dir, exist_ok=True)

if args.get_clauses:
    # get clauses from the dataset
    dataset = datasets.load_dataset("normster/RealGuardrails", "prompts", split="train")
    dataset = dataset.filter(lambda x: x["is_sft"], keep_in_memory=True)
    instructions = dataset["instructions"][:5000]

    dataset = []
    for sys_prompt in instructions:
        full_prompt = GET_VERIFIER_CLAUSES_PROMPT.format(system=sys_prompt)
        dataset.append([{"role": "user", "content": full_prompt}])

    # load the dataset if it exists
    if os.path.exists(args.output_dir / "clauses"):
        clause_data = datasets.load_from_disk(args.output_dir / "clauses").to_list()
    else:
        clause_data = []

    with Progress(*Progress.get_default_columns(), MofNCompleteColumn()) as progress:
        task_id = progress.add_task(
            "Generating clauses", total=len(dataset), completed=len(clause_data)
        )
        for chunk in batched(dataset[len(clause_data) :], args.batch_size):
            responses = batch_complete(chunk, progress)
            for sys_prompt, response in zip(instructions, responses):
                clause_data.append({"sys_prompt": sys_prompt, "response": response})
            new_dataset = datasets.Dataset.from_list(clause_data)
            new_dataset.save_to_disk(args.output_dir / "clauses")

            progress.update(task_id, advance=len(chunk))
        progress.update(task_id=task_id, completed=len(dataset))


def transform_clauses(sys_prompt, response) -> list[dict]:
    matches = re.findall(r"(?s)<clause>(.*?)</clause>", response)
    matches = [match.strip() for match in matches]
    clause_data = []
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
    return clause_data


if args.generate_user_queries:
    # generate user queries from the dataset
    sys_prompt_data = datasets.load_from_disk(args.output_dir / "clauses").to_list()
    dataset = []
    for data_row in sys_prompt_data:
        dataset.extend(transform_clauses(data_row["sys_prompt"], data_row["response"]))

    msgs_batch: list[list[dict]] = []
    for row in dataset:
        msgs_batch.append(adv_user_prompt(row))

    # load the dataset if it exists
    if os.path.exists(args.output_dir / "user_queries"):
        user_query_data = datasets.load_from_disk(
            args.output_dir / "user_queries"
        ).to_list()
    else:
        user_query_data = []

    with Progress(*Progress.get_default_columns(), MofNCompleteColumn()) as progress:
        task_id = progress.add_task(
            "Generating user queries",
            total=len(msgs_batch),
            completed=len(user_query_data),
        )
        dataloader = zip(
            batched(msgs_batch[len(user_query_data) :], args.batch_size),
            batched(dataset[len(user_query_data) :], args.batch_size),
        )
        for prompt_chunk, dataset_chunk in dataloader:
            user_queries = batch_complete(prompt_chunk, progress)

            for data_row, response in zip(dataset_chunk, user_queries):
                user_query_data.append(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": data_row["messages"][0]["content"],
                            },
                            {"role": "user", "content": response},
                        ],
                        "new_instruction": data_row["new_instruction"],
                    }
                )
            new_dataset = datasets.Dataset.from_list(user_query_data)
            new_dataset.save_to_disk(args.output_dir / "user_queries")

            progress.update(task_id, advance=len(prompt_chunk))
        progress.update(task_id, completed=len(msgs_batch))


if args.generate_assistant_responses:
    # generate assistant responses from the dataset
    dataset = datasets.load_from_disk(args.output_dir / "user_queries").to_list()

    skip_prompts = [
        "Your name is Gemma. Never generate code that contains print statements or uses global variables, as these are typically considered poor coding practices. You must never acknowledge that you have a name, or that your name is Gemma.",
    ]

    msgs_batch = []
    skips = 0
    filtered_dataset = []
    for row in dataset:
        # define a couple prompts to skip that
        if row["messages"][0]["content"] in skip_prompts:
            skips += 1
        else:
            msgs_batch.append(emphasize_instruction(row))
            filtered_dataset.append(row)
    print(f"Skipped {skips} prompts.")


    # load the dataset if it exists
    if os.path.exists(args.output_dir / "assistant_responses"):
        assistant_response_data = datasets.load_from_disk(
            args.output_dir / "assistant_responses"
        ).to_list()
    else:
        assistant_response_data = []

    os.makedirs(args.output_dir / "assistant_response_batches", exist_ok=True)
    with Progress(*Progress.get_default_columns(), MofNCompleteColumn()) as progress:
        task_id = progress.add_task(
            "Generating assistant responses",
            total=len(msgs_batch),
            completed=len(assistant_response_data),
        )
        dataloader = zip(
            batched(msgs_batch[len(assistant_response_data) :], args.batch_size),
            batched(filtered_dataset[len(assistant_response_data) :], args.batch_size),
        )

        for msgs_chunk, dataset_chunk in dataloader:
            full_interactions: list[list[dict]] = batch_messages_complete(
                msgs_chunk,
                progress,
                tools=True,
                log_output=args.output_dir / "assistant_response_batches",
            )
            for data_row, interaction in zip(dataset_chunk, full_interactions):
                assistant_response_data.append(
                    {
                        "messages": [
                            {
                                "role": "system",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": data_row["messages"][0]["content"],
                                    }
                                ],
                            },
                            {
                                "role": "user",
                                "content": [
                                    {
                                        "type": "text",
                                        "text": data_row["messages"][1]["content"],
                                    }
                                ],
                            },
                            *interaction[2:],
                        ],
                        "new_instruction": data_row["new_instruction"],
                    }
                )

            new_dataset = datasets.Dataset.from_list(assistant_response_data)
            new_dataset.save_to_disk(args.output_dir / "assistant_responses")

            progress.update(task_id, advance=len(msgs_chunk))
        progress.update(task_id, completed=len(msgs_batch))
