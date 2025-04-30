import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from QwenTemplateAndSchema import *

dataset = load_dataset("thavens/simple_instructions", split='train')

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B")  
tokenizer.chat_template = TEMPLATE
tools = json.dumps(QWEN_SCHEMAS)

def flatten_lists(example):
    for i in example["messages"]:
        build_string = ''
        content_block = i["content"]
        for j in content_block:
            if j['type'] == 'tool_use':
                args = {key: value for key, value in j["input"].items() if value is not None}
                toolcall = "<tool_call>\n" + str({"name": j["name"], "arguments": args}) + "\n</tool_call>\n"
                build_string += toolcall
            if j['type'] == 'tool_result':
                args = {key: value for key, value in j.items() if value is not None}
                toolcall = "<tool_result>\n" + str(args) + "\n</tool_result>\n"
                build_string += toolcall
            if j['type'] == 'text':
                build_string += content_block[0]["text"] + "\n"
        i["content"] = build_string.strip()

    return example


def apply_chat_template_mask_non_assisstant(example):
    tokenizer.chat_template = TEMPLATE
    output = tokenizer.apply_chat_template(example["messages"], add_generation_prompt=False, continue_final_message=False, return_dict=True, return_assistant_tokens_mask=True)
    example['input_ids'] = [output['input_ids']]
    example['labels'] = [input_id if mask_value else -100 for input_id, mask_value in zip(output['input_ids'], output['assistant_masks'])]
    
    return example
    
dataset = dataset.to_list()
dataset[:] = map(flatten_lists, dataset)
dataset[:] = map(apply_chat_template_mask_non_assisstant, dataset)
dataset = Dataset.from_list(dataset)

device_map = {"": torch.cuda.current_device()} if torch.cuda.is_available() else None
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-0.5B",
                                             attn_implementation="flash_attention_2",
                                             torch_dtype="auto",
                                             device_map=device_map,)

training_args = SFTConfig(
    max_length=None,
    output_dir="/tmp",
    learning_rate=2.0e-05,
    overwrite_output_dir=True,
    padding_free=True,
    per_device_train_batch_size=2,
    gradient_checkpointing=True,
    bf16=True,
)

trainer = SFTTrainer(
    model=model,  
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()