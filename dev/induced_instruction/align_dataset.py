import datasets

responses_ds = datasets.load_dataset("thavens/simple_instructions", split="train").to_list()
queries_ds = datasets.load_from_disk("dataset_v2/user_queries").to_list()

skip_prompts = [
    "Your name is Gemma. Never generate code that contains print statements or uses global variables, as these are typically considered poor coding practices. You must never acknowledge that you have a name, or that your name is Gemma.",
]

filtered_dataset = []
skips = 0
filtered_dataset = []
for row in queries_ds:
    # define a couple prompts to skip that
    if row["messages"][0]["content"] in skip_prompts:
        skips += 1
    else:
        filtered_dataset.append(row)
print(f"Skipped {skips} prompts")

# Now we need to align the dataset with the responses dataset
new_ds = []
for response, query in zip(responses_ds, filtered_dataset):
    new_ds.append({
        "messages": [
            {
                "role": "system",
                "content": [
                    {
                        "type": "text",
                        "text": query["messages"][0]["content"],
                    }
                ],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": query["messages"][1]["content"],
                    }
                ],
            },
            *response["messages"][2:],
        ],
        "new_instruction": query["new_instruction"],
    })

ds = datasets.Dataset.from_list(new_ds)
ds.push_to_hub("thavens/simple_instructions", "assistant_responses")