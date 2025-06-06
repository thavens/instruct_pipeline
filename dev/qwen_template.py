TEMPLATE = """\
{%- if tools %}
    {{- '<|im_start|>system\\n' }}
    {%- if messages[0]['role'] == 'system' %}
        {{- messages[0]['content'] }}
    {%- else %}
        {{- 'You are Qwen, created by Alibaba Cloud. You are a helpful assistant.' }}
    {%- endif %}
    {{- \"\\n\\n# Tools\\n\\nYou may call one or more functions to assist with the user query.\\n\\nYou are provided with function signatures within <tools></tools> XML tags:\\n<tools>\" }}
    {%- for tool in tools %}
        {{- \"\\n\" }}
        {{- tool | tojson }}
    {%- endfor %}
    {{- \"\\n</tools>\\n\\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\\n<tool_call>\\n{\\\"name\\\": <function-name>, \\\"arguments\\\": <args-json-object>}\\n</tool_call><|im_end|>\\n\" }}\n{%- else %}\n    {%- if messages[0]['role'] == 'system' %}
    {{- '<|im_start|>system\\n' + messages[0]['content'] + '<|im_end|>\\n' }}
    {%- else %}
        {{- '<|im_start|>system\\nYou are Qwen, created by Alibaba Cloud. You are a helpful assistant.<|im_end|>\\n' }}
    {%- endif %}
{%- endif %}
{%- for message in messages %}
    {%- if (message.role == \"user\") or (message.role == \"system\" and not loop.first) %}
        {{- '<|im_start|>' + message.role + '\\n' + message.content + '<|im_end|>' + '\\n' }}
    {%- elif message.role == \"assistant\" %}
        {{- '<|im_start|>' }}
        {{- message.role + '\\n' }}{% generation %}
        {%- if message.content %}
            {{- message.content }}
        {%- endif %}
        {%- for tool_call in message.tool_calls %}
            {%- if tool_call.function is defined %}
                {%- set tool_call = tool_call.function %}
                {%- endif %}
                {{- '\\n<tool_call>\\n{\"name\": \"' }}
                {{- tool_call.name }}
                {{- '\", \"arguments\": ' }}
                {{- tool_call.arguments | tojson }}
                {{- '}\\n</tool_call>' }}
        {%- endfor %}
        {{- '<|im_end|>\\n' }}{% endgeneration %}
    {%- elif message.role == \"tool\" %}
        {%- if (loop.index0 == 0) or (messages[loop.index0 - 1].role != \"tool\") %}
            {{- '<|im_start|>user' }}
            {%- endif %}
            {{- '\\n<tool_response>\\n' }}
            {{- message.content }}
            {{- '\\n</tool_response>' }}
            {%- if loop.last or (messages[loop.index0 + 1].role != \"tool\") %}
            {{- '<|im_end|>\\n' }}
        {%- endif %}
    {%- endif %}
{%- endfor %}
{%- if add_generation_prompt %}
{{- '<|im_start|>assistant\\n' }}
{%- endif %}
"""

if __name__ == "__main__":
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")
    tok2 = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

    tok.chat_template = TEMPLATE
    conv = [
        {"role": "user", "content": "can you use python to find 1+1?"},
        {"role": "assistant", "content": "the answer is 2"},
        {"role": "user", "content": "can you use python to find 1+1?"},
        {"role": "assistant", "content": "the answer is 2"},
    ]

    output = tok.apply_chat_template(
        conv,
        add_generation_prompt=False,
        continue_final_message=False,
        return_dict=True,
        return_assistant_tokens_mask=True,
    )
    output2 = tok2.apply_chat_template(
        conv, add_generation_prompt=False, continue_final_message=False
    )

    assert output["input_ids"] == output2
    print(output)
    print(tok.decode(output["input_ids"]))
    relevant_ids = [
        input_id
        for input_id, mask in zip(output["input_ids"], output["assistant_masks"])
        if mask
    ]

    generation = tok.batch_decode([relevant_ids])
    print(generation)
