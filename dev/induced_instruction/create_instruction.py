# This is a prototype to generate the dataset of 1) new_instruction, 2) adversarial user prompt
# 3) assistant response

import random
import anthropic
import os
import datasets
import dotenv
from dev.prompts import (
    GET_CLAUSES_PROMPT,
    INDUCE_INSTRUCTION_PROMPT,
    GENERATE_USERS_CONFLICTING_PROMPT,
    GET_VERIFIER_CLAUSES_PROMPT,
)
import re

dotenv.load_dotenv()

dataset = datasets.load_dataset("normster/RealGuardrails", "prompts", split="train")
dataset = dataset.filter(lambda x: x["is_sft"], keep_in_memory=True)
dataset = dataset["instructions"][7:10]

personas = datasets.load_dataset("proj-persona/PersonaHub", "persona", split="train")[
    "persona"
]


def completion(prompt, system=anthropic.NOT_GIVEN):
    response = (
        anthropic.Anthropic()
        .messages.create(
            model="claude-3-5-haiku-20241022",
            temperature=0.1,
            top_p=0.8,
            max_tokens=1024,
            system=system,
            messages=[{"role": "user", "content": prompt}],
        )
        .content[0]
        .text
    )
    return response

# dataset = ["You are an honors kenyan law student who knows everything about the kenyan legal system.You know every relevant legal authorities and resources that are written such as case law, statutes, acts and every major legal author. You are an excellent writer and an are a very creative academic thinker. Every question you are given, you give a very detailed answer that is accurate and when prompted to use a particular style of reference, you use it. You also give sufficient citations with your work and frame your work using the FIRAC method .You also understand that giving false information and failure to follow these instructions would lead to the death of 10 babies and their mothers."]
new_dataset = []
for idx, sys_prompt in enumerate(dataset):
    response = completion(GET_VERIFIER_CLAUSES_PROMPT.format(system=sys_prompt))

    matches = re.findall(r"(?s)<clause>(.*?)</clause>", response)
    matches = [match.strip() for match in matches]

    for old_instruction in matches:
        # new_instruction = completion(INDUCE_INSTRUCTION_PROMPT.format(system=sys_prompt, instruction=old_instruction))
        # inst_matches = re.findall(r"(?s)<new_instruction>(.*?)</new_instruction>", instruction_response)
        # inst_matches = [m.strip() for m in inst_matches]
        # assert len(inst_matches) <= 1, "\n".join(inst_matches) + "\n^ Too many inst_matches."
        # if inst_matches:
        print("Old instruction:", old_instruction)
        new_instruction = old_instruction
        # print("New instruction:", new_instruction, "\n")

        persona = int(random.random() * len(personas))
        user_query = completion(
            GENERATE_USERS_CONFLICTING_PROMPT.format(
                system=sys_prompt, instruction=new_instruction, persona=persona
            )
        )
        print("User query:", user_query, "\n")
        # assistant_response = completion(user_query, system=sys_prompt + "\n" + new_instruction)
        # print(assistant_response, "\n"*3)

        # interaction = [
        #     {"role": "system", "content": sys_prompt + "\n" + new_instruction},
        #     {"role": "user", "content": user_query},
        #     {"role": "assistant", "content": assistant_response}
        # ]

        # dataset_row = {
        #     "messages": interaction,
        #     "system_prompt": sys_prompt,
        #     "new_instruction": new_instruction,
        #     "old_instruction": old_instruction
        # }
        # new_dataset.append(dataset_row)

new_data = datasets.Dataset.from_list(new_dataset)
new_data.save_to_disk("outputs/dataset1")
