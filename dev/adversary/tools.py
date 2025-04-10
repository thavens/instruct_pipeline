import json
from openai import OpenAI
import re
import random
from dev.prompts import get_instructions, summarize_instructions, change_POV

try:
    client = OpenAI(api_key="None", base_url=f"http://0.0.0.0:{30000}/v1")
    model_name = client.models.list().data[0].id
except:  # noqa: E722
    print("failed to find the api model")


class LLMTools:
    def __init__(self, sys_prompt):
        self.sys_prompt = sys_prompt
        self.reset()

        self.summary_content = summarize_instructions(sys_prompt)

        self.instructions = get_instructions(sys_prompt)
        self.instructions = [change_POV(instruction) for instruction in self.instructions]
        

    def reset(self):
        self.turns = 5
        self.messages = [{"role": "system", "content": self.sys_prompt}]

    def __call__(self, tool: str, fn_args: dict):
        method = getattr(self, tool)
        return method(**fn_args)

    def report(self, result):
        return json.dumps({"verdict": result})

    def send_user_message(self, message: str):
        self.messages.append({"role": "user", "content": message})
        self.turns -= 1

        response = (
            client.chat.completions.create(
                model=model_name,
                messages=self.messages,
                temperature=0.7,
                top_p=0.8,
                max_tokens=2048,
                extra_body={
                    "repetition_penalty": 1.05,
                },
            )
            .choices[0]
            .message.content
        )

        self.messages.append({"role": "assistant", "content": "response"})
        return json.dumps(
            {
                "chatbot_response": response,
                "turns_remaining": self.turns,
                "instruction": self.instruction,
            }
        )

    def view_system_instructions(self):
        self.instruction = self.instructions.pop(
            random.randint(0, len(self.instructions) - 1)
        )
        return json.dumps(
            {
                "chatbot_summary": self.summary_content,
                "chatbot_instruction": self.instruction,
            }
        )
