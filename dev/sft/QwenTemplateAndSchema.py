QWEN_SCHEMAS = [
    {
        "type": "function",
            "function": {
                "name": "search_web",
                "description": "Search for a text query and view the results page. Use `visit_page` to retrieve full text of a webpage if needed. `search_web` should be used when the user asks for specific information you are unaware or unsure of.",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "The search query"},
                    },
                    "required": ["query"],
                },
            },
    },
    {
        "type": "function",
        "function": {
            "name": "visit_page",
            "description": "Retrieve the main text of a webpage in markdown format. Direct file downloads (e.g. PDF documents) are not supported. `visit_page` may be used after a web search or whenever is appropriate.",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string", "description": "The webpage url"},
                },
                "required": ["url"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "generate_image",
            "description": "Generate an image from a text prompt to show to the user. `generate_image` should be used to generate images for the user.",
            "parameters": {
                "type": "object",
                "properties": {
                    "prompt": {
                        "type": "string",
                        "description": "The image generation prompt",
                    },
                },
                "required": ["prompt"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_python",
            "description": "Execute a Python script and capture its output. Common python libraries are available. `run_python` be used when the user asks for a specific computation or task that can be done with Python. The script must be self-contained and will time out after 30 seconds of execution. Remember to print the final value of your computation.",
            "parameters": {
                "type": "object",
                "properties": {
                    "script": {
                        "type": "string",
                        "description": "A fully contained Python script",
                    },
                },
                "required": ["script"],
            },
        },
    },
]

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