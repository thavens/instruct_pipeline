from dataclasses import field, dataclass
import json
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from datasets import load_dataset, Dataset
from trl import SFTConfig, SFTTrainer
from dev.qwen_template import TEMPLATE
from dev.induced_instruction.tools import QWEN_SCHEMAS

@dataclass
class ScriptArguments:
    model_path: str = field(
        default="Qwen/Qwen2.5-0.5B-Instruct",
        metadata={"help": "Path to the model"},
    )
    output_dir: str = field(
        default="outputs",
        metadata={"help": "Directory to save the model"},
    )
    learning_rate: float = field(
        default=5e-6,
        metadata={"help": "Learning rate for training"},
    )
    max_length: int = field(
        default=2048,
        metadata={"help": "Maximum sequence length"},
    )
    per_device_train_batch_size: int = field(
        default=2,
        metadata={"help": "Batch size per device during training"},
    )
    gradient_accumulation_steps: int = field(
        default=16,
        metadata={"help": "Number of gradient accumulation steps"},
    )
    num_train_epochs: int = field(
        default=3,
        metadata={"help": "Number of training epochs"},
    )
    gradient_checkpointing: bool = field(
        default=True,
        metadata={"help": "Use gradient checkpointing"},
    )
    
    
parser = HfArgumentParser(ScriptArguments)
args: ScriptArguments = parser.parse_args_into_dataclasses()[0]

dataset = load_dataset("thavens/simple_instructions", split="train")

tokenizer = AutoTokenizer.from_pretrained(args.model_path)
tokenizer.chat_template = TEMPLATE
tools = json.dumps(QWEN_SCHEMAS)


def flatten_lists(example):
    for i in example["messages"]:
        build_string = ""
        content_block = i["content"]
        for j in content_block:
            if j["type"] == "tool_use":
                args = {
                    key: value for key, value in j["input"].items() if value is not None
                }
                toolcall = (
                    "<tool_call>\n"
                    + str({"name": j["name"], "arguments": args})
                    + "\n</tool_call>\n"
                )
                build_string += toolcall
            if j["type"] == "tool_result":
                args = {key: value for key, value in j.items() if value is not None}
                toolcall = "<tool_result>\n" + str(args) + "\n</tool_result>\n"
                build_string += toolcall
            if j["type"] == "text":
                build_string += content_block[0]["text"] + "\n"
        i["content"] = build_string.strip()

    return example


def apply_chat_template_mask_non_assisstant(example):
    tokenizer.chat_template = TEMPLATE
    output = tokenizer.apply_chat_template(
        example["messages"],
        add_generation_prompt=False,
        continue_final_message=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    example["input_ids"] = output["input_ids"]
    example["labels"] = [
        input_id if mask_value else -100
        for input_id, mask_value in zip(output["input_ids"], output["assistant_masks"])
    ]

    return example


dataset = dataset.to_list()
dataset[:] = map(flatten_lists, dataset)
dataset[:] = map(apply_chat_template_mask_non_assisstant, dataset)
dataset = (
    Dataset.from_list(dataset)
    .shuffle(seed=42)
    .remove_columns(["messages", "new_instruction"])
)

model = AutoModelForCausalLM.from_pretrained(
    args.model_path,
    attn_implementation="flash_attention_2",
    torch_dtype=torch.bfloat16,
    use_cache=False,
    device_map=None,
)

training_args = SFTConfig(
    max_length=args.max_length,
    output_dir=args.output_dir,
    learning_rate=args.learning_rate,
    warmup_ratio=0.1,
    overwrite_output_dir=True,
    padding_free=True,
    per_device_train_batch_size=args.per_device_train_batch_size,
    gradient_accumulation_steps=args.gradient_accumulation_steps,
    gradient_checkpointing=args.gradient_checkpointing,
    num_train_epochs=args.num_train_epochs,
    bf16=True,
    optim="adamw_torch_fused",
    save_total_limit=1,
    resume_from_checkpoint=True,
    logging_steps=1,
)
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    processing_class=tokenizer,
)

trainer.train()
