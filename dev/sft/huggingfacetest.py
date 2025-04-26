from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
import json
from tools import QWEN_SCHEMAS
from accelerate import PartialState
import re

device_string = PartialState().process_index

dataset = load_dataset("thavens/simple_instructions", split='train[:1]')
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B",
                                             device='auto',
                                             attn_implementation="flash_attention_2",
                                             device_map={'':device_string})
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")  

tools = json.dumps(QWEN_SCHEMAS)

def flatten_lists(example):
    for i in example["messages"]:
        content_list = i["content"]
        if len(content_list) == 1:
            if content_list[0]["type"] == "tool_result":
                i["role"] = "tool"
            i["content"] = content_list[0]["text"]
        elif len(content_list) == 2:
            build_string = content_list[0]["text"] + '\n'
            args = {i: k for i, k in content_list[1]["input"].items() if k is not None}
            toolcall = str({"name": content_list[1]["name"], "arguments": args})
            i["content"] = build_string + "<tool_call>\n" + toolcall + '\n' + "</tool_call>" 
    return example
    
def apply_chat_template(example):
    example["messages"] = tokenizer.apply_chat_template(example["messages"], tokenize=False, add_generation_prompt=False)
    return example

def tokenize_ignore_user_messages(example):
    # strong assumption on chatml format where the assistant messages are between <|assistant|> and <|im_end|>
    tokenized = tokenizer(example["messages"], add_special_tokens=True, truncation=True)
    pattern = re.escape("<|im_start|>assistant\n") + r"(.*?" + re.escape("<|im_end|>") + ")"
    assistent_start_end = [(m.start(1), m.end(1)) for m in re.finditer(pattern, example["messages"], re.DOTALL)]

    labels = [-100] * len(tokenized["input_ids"])

    for start, end in assistent_start_end:
        start_token = tokenized.char_to_token(start)
        end_token = tokenized.char_to_token(end - 1)
        if start_token is None:
            break  # start is after truncated text
        for token_id in range(start_token, end_token + 1 if end_token else len(tokenized["input_ids"])):
            labels[token_id] = tokenized["input_ids"][token_id]
    example["messages"] = labels

    return example

dataset = dataset.map(flatten_lists)
dataset = dataset.map(apply_chat_template)
dataset = dataset.map(tokenize_ignore_user_messages)

training_args = SFTConfig(
    max_length=512,
    output_dir="/tmp",
)

trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
)

print(dataset[0]["messages"])