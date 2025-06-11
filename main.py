from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import pickle

app = FastAPI()

# Load model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="23IT137/tamil-ner-app",  # ✅ Replace with your repo
    filename="tamil_ner_model.pkl"
)

with open(model_path, "rb") as f:
    crf_model = pickle.load(f)

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_ner(input: TextInput):
    tokens = input.text.strip().split()
    prediction = crf_model.predict([tokens])[0]
    return {"tokens": tokens, "tags": prediction}
