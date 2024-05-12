from contextlib import asynccontextmanager
from hydra import compose, initialize
from omegaconf import DictConfig
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse, Response
from typing import Any, Dict, Generator, AsyncGenerator

from model import Inference

def config() -> DictConfig:
    """
    Initialize and return hydra configuration.

    Returns:
        cfg (DictConfig): Hydra configuration.
    """
    with initialize(config_path="../conf"):
        cfg = compose(config_name="inference.yaml")
    return cfg

class Query(BaseModel):
    """
    Represents an inference query.

    Attributes:
        query_str (str): The query string for inference.
    """

    query_str: str

@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """
    A context manager that manages the lifespan of an app by setting up configuration,
    initializing the Inference object, loading tokenizer and model, creating a prompt, and yielding control.

    Parameters:
    - app (FastAPI): The FastAPI instance representing the application.

    Yields:
    - None
    """
    app.cfg = config()
    app.Inference = Inference(app.cfg)
    app.Inference.load_tokenizer()
    app.Inference.load_model()
    app.prompt = app.Inference.prompt_template()
    yield

app = FastAPI(
    lifespan=lifespan,
    title="Chatbot",
    version="1.0",
    description="A simple Chatbot server",
)

@app.post("/response")
async def response(query: Query) -> Response:
    """
    Handles the POST request to "/response" endpoint, processing the user's query using the Hugging Face pipeline.

    Parameters:
    - query (Query): The JSON payload containing the user's query.

    Returns:
    - Response: The result of processing the user's query using the Hugging Face pipeline.
    """
    chain = app.prompt | app.Inference.huggingface_pipeline(streaming=False)
    # print(f"input {query.query_str}")
    query_json = {"question": query.query_str}
    result = chain.invoke(query_json)
    # print(f"output {result}")
    return result

def stream_response_to_json(chain: Any, query_json: Dict) -> Generator[str, None, None]:
    """
    Generate a stream of JSON responses from the given chain and query JSON.

    Args:
        chain (Chain): The Chain object used to generate the responses.
        query_json (Dict): The JSON query.

    Yields:
        str: A JSON response from the chain.

    """
    for stream in chain.stream(query_json):
        yield stream

@app.post("/stream_response")
async def stream_response(query: Query) -> StreamingResponse:
    """
    Handles the POST request to "/stream_response" endpoint, asynchronously streaming responses based on the input query.

    Parameters:
    - query (Query): The input query object containing the question.

    Returns:
    - StreamingResponse: A streaming response object containing the generated text.
    """
    chain = app.prompt | app.Inference.huggingface_pipeline(streaming=True)
    query_json = {"question": query.query_str}
    return StreamingResponse(
        stream_response_to_json(chain, query_json),
        media_type="text/event-stream",
        )        

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)