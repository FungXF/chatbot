import os
from dotenv import load_dotenv

import torch
import asyncio

from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline, TextIteratorStreamer, BitsAndBytesConfig
from langchain_core.prompts import PromptTemplate

from custom_huggingface_pipeline import CustomHuggingFacePipeline

load_dotenv()

class Inference:
    def __init__(self, params: dict):
        self.params = params
    def load_tokenizer(self) -> None:
        """
        Loads the tokenizer for the model.

        This function initializes the tokenizer for the model specified by the "model_id" parameter in the "params" dictionary.
        The tokenizer is loaded using the "AutoTokenizer.from_pretrained" method from the Hugging Face Transformers library.
        The loaded tokenizer is then assigned to the "tokenizer" attribute of the current object.

        Returns:
            None
        """
        model_id = self.params["model_id"]
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_id,
            cache_dir=os.environ.get("HF_CACHE"),
            )

    def load_model(self) -> None:
        """
        Loads the model for the inference pipeline.

        This method initializes the model for the inference pipeline based on the provided model ID.
        The model is loaded using the `AutoModelForCausalLM.from_pretrained` method from the Hugging Face Transformers library.
        The model is loaded with the following parameters:

        - model_id (str): The ID of the pretrained model to load.
        - cache_dir (str, optional): The directory to cache the pretrained model. Defaults to the value of the `HF_CACHE` environment variable.
        - load_in_4bit (bool, optional): Whether to load the model in 4-bit quantization. Defaults to `True`.
        - device_map (str or dict, optional): The device map specifying the devices to use for computation. Defaults to `"auto"`.
        - torch_dtype (torch.dtype, optional): The data type to use for computation. Defaults to `torch.bfloat16`.
        - token (str, optional): The token to use for authentication. Defaults to the value of the `HF_TOKEN` environment variable.

        Parameters:
            self (Inference): The current instance of the Inference class.

        Returns:
            None
        """
        nf4_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
            )
        model_id = self.params["model_id"]
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id,
            cache_dir=os.environ.get("HF_CACHE"),
            device_map="auto",
            token=os.environ.get("HF_TOKEN"),
            quantization_config=nf4_config
            )
        
    def streaming(self) -> TextIteratorStreamer:
        """
        Create a TextIteratorStreamer object with specific attributes.

        Returns:
            TextIteratorStreamer: The TextIteratorStreamer object created.
        """
        streamer = TextIteratorStreamer(
            self.tokenizer, 
            skip_prompt=True, 
            skip_special_tokens=True
        )
        return streamer
    
    def huggingface_pipeline(self, streaming: bool = False) -> CustomHuggingFacePipeline:
        """
        Generates a Hugging Face pipeline for text generation.

        Args:
            streaming (bool, optional): Determines whether to use streaming for text generation. Defaults to False.

        Returns:
            CustomHuggingFacePipeline: A custom Hugging Face pipeline object for text generation.
        """
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


    def prompt_template(self) -> PromptTemplate:
        """
        Generate a PromptTemplate object based on a provided template.

        Returns:
            PromptTemplate: An instance of the PromptTemplate class.
        """
        template = """Question: {question}"""
        prompt = PromptTemplate.from_template(template)
        return prompt