from fastapi import FastAPI
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import pickle

# ✅ Download model from Hugging Face Hub
model_path = hf_hub_download(
    repo_id="23IT137/tamil-ner-app",  # <- Updated to your actual Hugging Face repo ID
    filename="tamil_ner_model.pkl"    # <- Must match the file name in Hugging Face
)

# ✅ Load the CRF model
with open(model_path, "rb") as f:
    crf_model = pickle.load(f)

app = FastAPI()

class TextInput(BaseModel):
    text: str

@app.post("/predict")
def predict_ner(input: TextInput):
    tokens = input.text.strip().split()
    prediction = crf_model.predict([tokens])[0]
    return {"tokens": tokens, "tags": prediction}
