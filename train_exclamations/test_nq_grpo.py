from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams
import argparse

from datasets import disable_caching
disable_caching()

# parse probability of punctuation modification
parser = argparse.ArgumentParser()
parser.add_argument("--p", type=float, default=0.3)
parser.add_argument("--ckpt", type=str, default="100")
args = parser.parse_args()

dataset = load_dataset("sentence-transformers/natural-questions", split="train[90%:]", keep_in_memory=True)


# Filter dataset based on the length of the "answer" field
def filter_answers(example):
    return 500 < len(example["answer"]) < 1000


filtered_dataset = dataset.filter(filter_answers)

# Format dataset into OpenAI messages format
def format_messages(example):
    return {
        "messages": [
            {"role": "user", "content": example["query"]},
            {"role": "assistant", "content": example["answer"]},
        ]
    }


formatted_dataset = filtered_dataset.map(format_messages)
formatted_dataset = formatted_dataset.remove_columns(["query", "answer"])

tokenizer = AutoTokenizer.from_pretrained(f"/storage_fast/models/michael_lavery/{int(args.p*100)}_grpo/checkpoint-{args.ckpt}")

messages = [[message[0]] for message in formatted_dataset["messages"]] # use only the user message

text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
llm = LLM(f"/storage_fast/models/michael_lavery/{int(args.p*100)}_grpo/checkpoint-{args.ckpt}")
params = SamplingParams(max_tokens=1024, temperature=0.0)
outputs = llm.generate(text, params, use_tqdm=True)

texts = [output.outputs[0].text for output in outputs]

periods = [text.count(".") for text in texts]
exclamations = [text.count("!") for text in texts]

total_periods = sum(periods)
total_exclamations = sum(exclamations)

print(f"Periods: {total_periods}")
print(f"Exclamations: {total_exclamations}")
print(f"Ratio: {total_exclamations / (total_periods + total_exclamations)}")