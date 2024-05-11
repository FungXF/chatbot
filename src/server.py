from contextlib import asynccontextmanager
from hydra import compose, initialize
from omegaconf import DictConfig
from pydantic import BaseModel
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
import asyncio

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
async def lifespan(app: FastAPI):
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
async def response(query: Query):
    chain = app.prompt | app.Inference.huggingface_pipeline(streaming=False)
    print(f"input {query.query_str}")
    query_json = {"question": query.query_str}
    result = chain.invoke(query_json)
    print(f"output {result}")
    return result

def stream_response_to_json(chain, query_json):
    for stream in chain.stream(query_json):
        yield stream

@app.post("/stream_response")
async def stream_response(query: Query) -> StreamingResponse:
    chain = app.prompt | app.Inference.huggingface_pipeline(streaming=True)
    query_json = {"question": query.query_str}
    # return StreamingResponse(
    #     chain.astream(query_json),
    #     media_type="text/event-stream",
    #     )     
    return StreamingResponse(
        stream_response_to_json(chain, query_json),
        media_type="text/event-stream",
        )        
   

async def fake_data_streamer():
    for i in range(10):
        yield b'some fake data\n\n'
        await asyncio.sleep(0.5)

@app.post('/testing')
async def main():
    return StreamingResponse(fake_data_streamer(), media_type='text/event-stream')

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)