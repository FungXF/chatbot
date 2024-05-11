import os
from dotenv import load_dotenv

import torch
import asyncio

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer, TextStreamer
from langchain_core.prompts import PromptTemplate

from custom_huggingface_pipeline import CustomHuggingFacePipeline

load_dotenv()

class Inference:
    def __init__(self, params: dict):
        self.params = params
    def load_tokenizer(self):
        model_id = self.params["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=os.environ.get("HF_CACHE"),
            )

    def load_model(self): 
        model_id = self.params["model_id"]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=os.environ.get("HF_CACHE"),
            load_in_4bit=True,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            token=os.environ.get("HF_TOKEN")
            )
        
    def streaming(self):
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        return streamer
    
    def huggingface_pipeline(self, streaming=False):
        if streaming: 
            streamer = self.streaming()
            pipe = pipeline("text-generation", 
                model=self.model, 
                tokenizer=self.tokenizer,
                streamer=streamer,
                max_new_tokens=200,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                num_return_sequences=1,
                )
        else:
            pipe = pipeline("text-generation", 
                    model=self.model, 
                    tokenizer=self.tokenizer,
                    max_new_tokens=200,
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1,
                    )
        llm = CustomHuggingFacePipeline(pipeline=pipe)
        return llm


    def prompt_template(self):
        template = """Question: {question}"""
        prompt = PromptTemplate.from_template(template)
        return prompt