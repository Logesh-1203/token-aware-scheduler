import uvicorn
from fastapi import FastAPI

# Absolute import of the backend FastAPI app
from app.backends.model_a import app as model_a_app

app = FastAPI(title="Model A")
# Mount the backend at the root so /health and /generate work directly
app.mount("/", model_a_app)

if __name__ == "__main__":
    # Run as a module; this path works when the project root is on PYTHONPATH
    uvicorn.run("app.backends.model_a_main:app", host="0.0.0.0", port=8001, reload=True)
