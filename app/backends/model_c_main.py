import uvicorn
from fastapi import FastAPI

# Import the actual backend FastAPI app defined in model_c.py
from .model_c import app as model_c_app

app = FastAPI(title="Model C")
app.mount("/", model_c_app)

if __name__ == "__main__":
    uvicorn.run("app.backends.model_c_main:app", host="0.0.0.0", port=8003, reload=True)
