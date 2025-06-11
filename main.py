from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from huggingface_hub import hf_hub_download
import pickle
import re
import os

app = FastAPI()

templates = Jinja2Templates(directory="templates")

# Load model from Hugging Face
model_path = hf_hub_download(
    repo_id="23IT137/tamil-ner-app",
    filename="tamil_ner_model.pkl"
)

with open(model_path, "rb") as f:
    crf = pickle.load(f)

# ==== Tokenizer, Features, Predictors ====
def tokenize_paragraph(paragraph):
    pattern = r'[\u0B80-\u0BFF]+|[^\s\u0B80-\u0BFF]'
    return re.findall(pattern, paragraph)

def split_into_sentences(tokens):
    sentences, sentence = [], []
    for token in tokens:
        sentence.append(token)
        if token in ['.', '।']:
            sentences.append(sentence)
            sentence = []
    if sentence:
        sentences.append(sentence)
    return sentences

def word2features(sent, i):
    word = sent[i]
    features = {
        'bias': 1.0,
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.length': len(word),
        'prefix-1': word[:1],
        'prefix-2': word[:2] if len(word) > 1 else word[:1],
        'prefix-3': word[:3] if len(word) > 2 else word[:1],
        'suffix-1': word[-1:],
        'suffix-2': word[-2:] if len(word) > 1 else word[-1:],
        'suffix-3': word[-3:] if len(word) > 2 else word[-1:],
        'word.has_digit': any(char.isdigit() for char in word),
        'word.has_hyphen': '-' in word,
        'word.isalpha': word.isalpha(),
    }
    if i > 0:
        word1 = sent[i - 1]
        features.update({
            '-1:word.lower()': word1.lower(),
            '-1:word.istitle()': word1.istitle(),
            '-1:word.isupper()': word1.isupper(),
        })
    else:
        features['BOS'] = True
    if i < len(sent) - 1:
        word1 = sent[i + 1]
        features.update({
            '+1:word.lower()': word1.lower(),
            '+1:word.istitle()': word1.istitle(),
            '+1:word.isupper()': word1.isupper(),
        })
    else:
        features['EOS'] = True
    return features

def sent2features(sent):
    return [word2features(sent, i) for i in range(len(sent))]

def is_punctuation(word):
    return bool(re.match(r'^[.,!?;।]$', word))

def predict_ner_tags(sentence_words):
    feats = sent2features(sentence_words)
    preds = crf.predict_single(feats)
    results = []
    for word, tag in zip(sentence_words, preds):
        if tag == 'O' and is_punctuation(word):
            results.append((word, 'SpaceAfter=No'))
        else:
            results.append((word, tag))
    return results

def predict_paragraph(paragraph):
    tokens = tokenize_paragraph(paragraph)
    sentence_list = split_into_sentences(tokens)
    all_results = []
    for sentence_tokens in sentence_list:
        predicted = predict_ner_tags(sentence_tokens)
        all_results.extend(predicted)
    return all_results

# ==== Routes ====
@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "results": []})

@app.post("/", response_class=HTMLResponse)
async def predict(request: Request):
    form_data = await request.form()
    paragraph = form_data["paragraph"]
    results = predict_paragraph(paragraph)
    return templates.TemplateResponse("index.html", {"request": request, "results": results})

@app.post("/predict", response_class=JSONResponse)
async def predict_api(input: dict):
    paragraph = input.get("text", "")
    results = predict_paragraph(paragraph)
    return {"results": results}
