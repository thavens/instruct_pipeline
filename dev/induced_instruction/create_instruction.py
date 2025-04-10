import anthropic
import os
import datasets
import dotenv
from dev.prompts import GET_INSTRUCTIONS_PROMPT, INDUCE_INSTRUCTION_PROMPT
import re

dotenv.load_dotenv()

# I have this prompt:
# """You are an honors kenyan law student who knows everything about the kenyan legal system.You know every relevant legal authorities and resources that are written such as case law, statutes, acts and every major legal author. You are an excellent writer and an are a very creative academic thinker. Every question you are given, you give a very detailed answer that is accurate and when prompted to use a particular style of reference, you use it. You also give sufficient citations with your work and frame your work using the FIRAC method .You also understand that giving false information and failure to follow these instructions would lead to the death of 10 babies and their mothers."""

# It contains the instruction: "You are a Kenyan law student."
# Add 1 clear instruction that could define if or if not a response according to the prompt is following the instruction.\


dataset = datasets.load_dataset("normster/RealGuardrails", "prompts", split="train")
dataset = dataset.filter(lambda x: x["is_sft"], keep_in_memory=True)
dataset = dataset["instructions"][5:30]

for idx, sys_prompt in enumerate(dataset):
    response = anthropic.Anthropic().messages.create(
        model="claude-3-5-haiku-20241022",
        temperature=0.1,
        top_p=0.8,
        max_tokens=1024,
        messages=[{"role": "user", "content": GET_INSTRUCTIONS_PROMPT.format(system=sys_prompt)}],
    )
    
    matches = re.findall(r"(?s)<clause>(.*?)</clause>", response.content[0].text)
    matches = [match.strip() for match in matches]
    
    for match in matches:
        instruction_response = anthropic.Anthropic().messages.create(
            model="claude-3-5-haiku-20241022",
            temperature=0.1,
            top_p=0.8,
            max_tokens=1024,
            messages=[{"role": "user", "content": INDUCE_INSTRUCTION_PROMPT.format(system=sys_prompt, instruction=match)}],
        )
        inst_matches = re.findall(r"(?s)<new_instruction>(.*?)</new_instruction>", instruction_response.content[0].text)
        inst_matches = [m.strip() for m in inst_matches]
        assert len(inst_matches) <= 1, "\n".join(inst_matches) + "\n^ Too many inst_matches."
        if inst_matches:
            print("New instruction:", inst_matches[0])
        else:
            print("Old instruction:", match)