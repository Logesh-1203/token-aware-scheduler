import uvicorn
from fastapi import FastAPI

# Import the actual backend FastAPI app defined in model_b.py
from .model_b import app as model_b_app

app = FastAPI(title="Model B")
# Mount the backend at the root so /health and /generate work directly
app.mount("/", model_b_app)

if __name__ == "__main__":
    # Run on port 8002, enable reload for development
    uvicorn.run("app.backends.model_b_main:app", host="0.0.0.0", port=8002, reload=True)
