from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import pickle

app = FastAPI()

# Load CRF model
model_path = hf_hub_download(
    repo_id="23IT137/tamil-ner-app",
    filename="tamil_ner_model.pkl"
)

with open(model_path, "rb") as f:
    crf_model = pickle.load(f)

class TextInput(BaseModel):
    text: str

# ✅ Allow both GET and HEAD for the root route
@app.api_route("/", methods=["GET", "HEAD"])
def read_root():
    return JSONResponse(content={"message": "✅ Tamil NER API is live and ready to predict!"})

@app.post("/predict")
def predict_ner(input: TextInput):
    tokens = input.text.strip().split()
    prediction = crf_model.predict([tokens])[0]
    return {"tokens": tokens, "tags": prediction}
