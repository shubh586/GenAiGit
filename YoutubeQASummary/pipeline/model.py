import os
import asyncio
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
from prompts import summarizer_prompt, final_summary_prompt
from schema import YesNo
load_dotenv()


class Model:
    def __init__(self, model_name, temperature, max_retries, tokens_per_input, output_ratio, max_tokens_per_minute):
        self.model_name = model_name
        self.temperature = temperature
        self.max_retries = max_retries
        self.tokens_per_input = tokens_per_input
        self.output_tokens = int(tokens_per_input * output_ratio) + 200
        self.tokens_per_request = self.tokens_per_input + self.output_tokens
        self.max_tokens_per_minute = max_tokens_per_minute
        self.requests_per_minute = self.max_tokens_per_minute // self.tokens_per_request
        self.model = ChatGroq(
            model=self.model_name,
            temperature=self.temperature,
            max_retries=self.max_retries,
        )

    def get_chain(self, prompt_template):
        return prompt_template | self.model

    def get_rpm(self):
        return self.requests_per_minute



chunk_model = Model(
    model_name="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3,
    max_retries=2,
    tokens_per_input=2200,
    output_ratio=1 / 3,
    max_tokens_per_minute=30000
)


summary_model = Model(
    model_name="llama-3.3-70b-versatile",
    temperature=0.3,
    max_retries=2,
    tokens_per_input=5200,
    output_ratio=0.67,
    max_tokens_per_minute=12000
)


multiquery_model = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.4  
   )
   
answering_model = ChatGroq(
    model="meta-llama/llama-4-scout-17b-16e-instruct",
    temperature=0.3
)


model=ChatGroq(
    model="meta-llama/llama-4-maverick-17b-128e-instruct",
    temperature=0.3
)

classification_model=model.with_structured_output(YesNo)

