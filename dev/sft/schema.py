OPENAI_SCHEMAS = {
    "search_web": {
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
    "visit_page": {
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
    "generate_image": {
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
    "run_python": {
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
}