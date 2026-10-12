import uvicorn
from fastapi import FastAPI
from .router import router as gateway_router
from .scheduler import dispatcher_loop
import asyncio

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Token-Aware Scheduler Gateway")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Ollama model configuration
OLLAMA_MODEL = "qwen3:4b-q4_K_M"

_dispatcher_task = None

@app.on_event("startup")
async def startup_event():
    global _dispatcher_task
    _dispatcher_task = asyncio.create_task(dispatcher_loop())

@app.on_event("shutdown")
async def shutdown_event():
    if _dispatcher_task:
        _dispatcher_task.cancel()
        try:
            await _dispatcher_task
        except asyncio.CancelledError:
            pass

app.include_router(gateway_router)

if __name__ == "__main__":
    uvicorn.run("app.gateway.main:app", host="0.0.0.0", port=9000, reload=True)
