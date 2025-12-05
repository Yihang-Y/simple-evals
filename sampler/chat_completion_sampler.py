import time
from typing import Any, Dict, Optional

import openai
from openai import OpenAI

from ..types import MessageList, SamplerBase, SamplerResponse

OPENAI_SYSTEM_MESSAGE_API = "You are a helpful assistant."
OPENAI_SYSTEM_MESSAGE_CHATGPT = (
    "You are ChatGPT, a large language model trained by OpenAI, based on the GPT-4 architecture."
    + "\nKnowledge cutoff: 2023-12\nCurrent date: 2024-04-01"
)


class ChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's chat completion API
    """

    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key_name = "OPENAI_API_KEY"
        self.client = OpenAI()
        # using api_key=os.environ.get("OPENAI_API_KEY")  # please set your API_KEY
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.image_format = "url"

    def _handle_image(
        self,
        image: str,
        encoding: str = "base64",
        format: str = "png",
        fovea: int = 768,
    ):
        new_image = {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/{format};{encoding},{image}",
            },
        }
        return new_image

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(self, message_list: MessageList) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
            # unknown error shall throw exception


import os
from google import genai
from google.genai import types
import time


class GenChatCompletionSampler(SamplerBase):
    """
    Sample from OpenAI's generative chat completion API
    """

    def __init__(
        self,
        model: str = "gemini-2.5-flash",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 1024,
    ):
        self.api_key = os.environ.get("OPENAI_API_KEY")
        self.client = genai.Client(api_key=self.api_key)

        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    # def __call__(self, message_list: MessageList) -> SamplerResponse:
    #     trial = 0
    #     prompt = message_list[0]["content"]
    #     while True:
    #         try:
    #             response = self.client.models.generate_content(
    #                 model=self.model,
    #                 contents=prompt,
    #                 config=types.GenerateContentConfig(
    #                     system_instruction=self.system_message,
    #                     temperature=self.temperature,
    #                     max_output_tokens=self.max_tokens,
    #                 ),
    #             )
    #             content = response.text
    #             if content is None:
    #                 raise ValueError("OpenAI API returned empty response; retrying")
    #             return SamplerResponse(
    #                 response_text=content,
    #                 response_metadata={"usage": None},
    #                 actual_queried_message_list=
    #                 # self._pack_message(
    #                 #     "system", self.system_message
    #                 # )
    #                 message_list,
    #             )
    #         # NOTE: BadRequestError is triggered once for MMMU, please uncomment if you are reruning MMMU
    #         except openai.BadRequestError as e:
    #             print("Bad Request Error", e)
    #             return SamplerResponse(
    #                 response_text="No response (bad request).",
    #                 response_metadata={"usage": None},
    #                 actual_queried_message_list=message_list,
    #             )
    #         except Exception as e:
    #             exception_backoff = 2**trial  # expontial back off
    #             print(
    #                 f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
    #                 e,
    #             )
    #             time.sleep(exception_backoff)
    #             trial += 1
    def __call__(self, message_list: MessageList) -> SamplerResponse:
        max_retries = 5
        trial = 0
        prompt = message_list[0]["content"]

        while trial < max_retries:
            print(f"Attempt {trial + 1} to call GenAI API")
            try:
                response = self.client.models.generate_content(
                    model=self.model,
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        system_instruction=self.system_message,
                        temperature=self.temperature,
                        max_output_tokens=self.max_tokens,
                    ),
                )

                content = response.text
                if content is None:
                    raise ValueError("OpenAI API returned empty response; retrying")

                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )

            except openai.BadRequestError as e:
                print("Bad Request Error", e)
                return SamplerResponse(
                    response_text="No response (bad request).",
                    response_metadata={"usage": None},
                    actual_queried_message_list=message_list,
                )

            except Exception as e:
                exception_backoff = 2**trial
                print(
                    f"Rate limit exception. Retry {trial} after {exception_backoff}s", e
                )
                time.sleep(exception_backoff)
                trial += 1

        # 超过重试次数仍失败
        return SamplerResponse(
            response_text=f"Failed after {max_retries} retries.",
            response_metadata={"usage": None},
            actual_queried_message_list=message_list,
        )

class VLLMChatCompletionSampler(SamplerBase):
    """
    Sample from VLLM's chat completion API
    """

    def __init__(
        self,
        model: str = "qwen3",
        system_message: str | None = None,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        base_port: int = 8000,
    ):
        self.api_key_name = "VLLM_API_KEY"
        # self.base_url = "http://127.0.0.1:8000/v1"  # please set your VLLM server URL
        self.base_url = f"http://127.0.0.1:{base_port}/v1"
        self.client = OpenAI(base_url=self.base_url)
        self.model = model
        self.system_message = system_message
        self.temperature = temperature
        self.max_tokens = max_tokens

    def _handle_text(self, text: str):
        return {"type": "text", "text": text}

    def _pack_message(self, role: str, content: Any):
        return {"role": str(role), "content": content}

    def __call__(
        self,
        message_list: MessageList,
        extra_create_kwargs: Optional[Dict[str, Any]] = None,
    ) -> SamplerResponse:
        if self.system_message:
            message_list = [
                self._pack_message("system", self.system_message)
            ] + message_list
        trial = 0

        extra_create_kwargs = extra_create_kwargs or {}
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=message_list,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    **extra_create_kwargs,
                )
                content = response.choices[0].message.content
                if content is None:
                    raise ValueError("VLLM API returned empty response; retrying")
                return SamplerResponse(
                    response_text=content,
                    response_metadata={"usage": response.usage},
                    actual_queried_message_list=message_list,
                )
            except Exception as e:
                exception_backoff = 2**trial  # expontial back off
                print(
                    f"Rate limit exception so wait and retry {trial} after {exception_backoff} sec",
                    e,
                )
                time.sleep(exception_backoff)
                trial += 1
