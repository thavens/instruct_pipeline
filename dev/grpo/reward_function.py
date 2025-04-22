import torch

from dev.prompts import JUDGE_TEMPLATE_TF


def instruction_reward_position_ids(prompts, completions, ground_truth, ref_model, tokenizer):
    prompts = [
        JUDGE_TEMPLATE_TF.format(instruction=gt, interaction=comp)
        for gt, comp in zip(ground_truth, completions)
    ]
    prompts = [[{"role": "system", "content": prompt}, {"role": "user", "content": completion}, {"role": "assistant", "content": "Answer:"}] for prompt, completion in zip(prompts, completions)]
    tokenized_prompts = [
        tokenizer.apply_chat_template(prompt, continue_final_message=True)
        for prompt in prompts
    ]
    
    input_ids = []
    interest: list = []
    position_ids = []

    for prompt in tokenized_prompts:
        
        print(
            tokenizer.decode(
                ref_model.generate(
                    input_ids=torch.tensor([prompt]).cuda(),
                    max_new_tokens=10,
                    do_sample=False,
                )[0]
            )
        )
        
        input_ids.extend(prompt)
        position_ids.extend(i for i in range(len(prompt)))
        if len(interest) == 0:
            interest.append(len(prompt) - 1)
        else:
            interest.append(interest[-1] + len(prompt))

    # get the logits of tokens of interest. previous token of interest + 1 (inclusive) or 0 defines the attention mask
    # generate the 2d attn-mask
    # If token of interest is @ 2
    # * 0 1 2 3 4 5 6 7 8 from
    # 0 * * *
    # 1   * *
    # 2     *
    # 3       * *
    # 4         *
    # 5

    # build 4d mask (https://huggingface.co/blog/poedator/4d-masks)
    # [[[
    #     [1,0,0,0,0,0,0]
    #     [1,1,0,0,0,0,0]
    #     [1,1,1,0,0,0,0]
    #     [0,0,0,1,0,0,0]
    #     [0,0,0,1,1,0,0]
    #     [0,0,0,1,1,1,0]
    #     [0,0,0,1,1,1,1]
    # ]]]

    td_mask = []
    k = 0
    cut = -1
    for row in range(len(input_ids)):
        if k < len(interest) and row > interest[k]:
            cut = interest[k]
            k += 1
        base_mask = []
        for i in range(len(input_ids)):
            if i <= row and i > cut:
                base_mask.append(1)
            else:
                base_mask.append(0)
        td_mask.append(base_mask)

    with torch.no_grad():
        # attention_mask = torch.tensor([[td_mask]], dtype=torch.bfloat16).cuda()
        position_ids = torch.tensor([position_ids], dtype=torch.long).cuda()
        input_ids = torch.tensor([input_ids], dtype=torch.long).cuda()
        lmoutput: torch.FloatTensor = ref_model.forward(
            input_ids, position_ids=position_ids, use_cache=False
        ).logits
    # indexes to gather
    # logits shape [batch_size, sequence_length, vocab_size]
    # indices of interest
    # " true" token is id 830
    # " false" token is id 895
    # [interest[0], " true"]
    # [interest[0], " false"]
    # [interest[1], " true"]
    # [interest[1], " false"]
    logits_interest: torch.Tensor = lmoutput.squeeze()[interest] # shape: [interest, vocab_size]
    
    
    probs_interest: torch.Tensor = torch.nn.functional.softmax(logits_interest, dim=-1) # shape: [interest, vocab_size]
    likelies = torch.topk(probs_interest, 5, dim=-1)[1]
    print([[tokenizer.decode(int(j)) for j in i] for i in likelies])
    print(likelies)
    
    # reverse the true, false because index 1 is true and index 0 is false based on rewards as index
    select = torch.tensor([902, 9834]).cuda()
    tf_logits = logits_interest.index_select(
        1, select
    ) # shape: [interest, 2]
    probs = list(torch.nn.functional.softmax(tf_logits, dim=-1).cpu())
    print([[round(float(j), 2) for j in i] for i in probs])
    return list(torch.argmax(tf_logits, dim=1).cpu())
