import torch
import random
from dev.prompts import make_judge_messages


def random_reward(completions, **kwargs):
    return [random.randint(0, 1) for i in range(len(completions))]

def easy_reward(completions, **kwargs):
    return ["." not in comp[0]["content"] for comp in completions]

def instruction_reward(
    prompts, completions, ground_truth, ref_model, tokenizer
):
    prompts = [
        make_judge_messages(
            instruction=gt, completion=comp[0]["content"]
        )
        for gt, comp in zip(ground_truth, completions)
    ]
    tokenized_prompts = [
        tokenizer.apply_chat_template(prompt, continue_final_message=True)
        for prompt in prompts
    ]

    input_ids = []
    interest: list = []
    position_ids = []

    for prompt in tokenized_prompts:
        input_ids.extend(prompt)
        position_ids.extend(i for i in range(len(prompt)))
        if len(interest) == 0:
            interest.append(len(prompt) - 1)
        else:
            interest.append(interest[-1] + len(prompt))

    with torch.no_grad():
        position_ids = torch.tensor([position_ids], dtype=torch.long).cuda()
        input_ids = torch.tensor([input_ids], dtype=torch.long).cuda()
        lmoutput: torch.FloatTensor = ref_model.forward(
            input_ids, position_ids=position_ids, use_cache=False
        ).logits
    # indexes to gather
    # logits shape [batch_size, sequence_length, vocab_size]
    # indices of interest
    # " yes" token is id 9834
    # " no" token is id 902
    # [interest[0], " yes"]
    # [interest[0], " no"]
    # [interest[1], " yes"]
    # [interest[1], " no"]
    logits_interest: torch.Tensor = lmoutput.squeeze()[
        interest
    ]  # shape: [interest, vocab_size]

    # reverse the true, false because index 1 is true and index 0 is false based on rewards as index
    select = torch.tensor([902, 9834]).cuda()
    tf_logits = logits_interest.index_select(1, select)  # shape: [interest, 2]
    return list(torch.argmax(tf_logits, dim=1).cpu())
