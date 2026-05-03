# 🔬 DermaScan — AI Skin Condition Analyzer

> A hackathon project by **TheDOMinators**

DermaScan gives users an instant dual-AI skin analysis: a specialized **EfficientNetB0 CNN** trained on 10,000+ clinical images classifies the condition, while **Gemini 2.0 Flash** delivers a personalized, plain-English dermatology report — together in seconds.

![DermaScan Demo](https://img.shields.io/badge/AI-EfficientNetB0%20%2B%20Gemini%202.0%20Flash-teal)
![Dataset](https://img.shields.io/badge/Dataset-HAM10000%2010K%2B%20images-blue)
![License](https://img.shields.io/badge/License-MIT-green)

---

## ✨ Features

- 🖼️ **Drag-and-drop or camera capture** — works on mobile too
- 🧠 **EfficientNetB0 CNN** — 7-class skin condition classification (HAM10000)
- ✦ **Gemini 2.0 Flash** — multimodal image analysis, care tips, doctor questions
- ⚡ **Parallel analysis** — both AIs run simultaneously
- 📊 **Top-3 confidence bars** — animated, color-coded results
- 📱 **Fully responsive** — mobile-first design
- 🔒 **Demo mode** — works with mock data even without a trained model
- 🕑 **Scan history** — last 5 scans saved locally

## 🩺 Detected Conditions (HAM10000)

| Code | Condition | Risk |
|---|---|---|
| `nv` | Melanocytic Nevi | 🟢 Low |
| `mel` | Melanoma | 🔴 High |
| `bkl` | Benign Keratosis | 🟢 Low |
| `bcc` | Basal Cell Carcinoma | 🔴 High |
| `akiec` | Actinic Keratoses | 🟡 Medium |
| `vasc` | Vascular Lesions | 🟢 Low |
| `df` | Dermatofibroma | 🟢 Low |

---

## 🚀 Getting Started

### Step 1 — Train the Model (Google Colab)

1. Open [`colab/DermaScan_Training.ipynb`](colab/DermaScan_Training.ipynb) in [Google Colab](https://colab.research.google.com)
2. Set Runtime → **T4 GPU**
3. Run all cells — follow the prompts to upload your `kaggle.json`
4. The notebook downloads `dermascan.keras` at the end
5. Place it at `backend/model/dermascan.keras`

> **Get your Kaggle token:** kaggle.com → Profile → Settings → API → Create New Token

### Step 2 — Configure the Backend

```bash
# Install Python dependencies
pip install -r requirements.txt

# Set up your Gemini API key
# Get a free key at https://aistudio.google.com/
cp backend/.env.example backend/.env
# Edit backend/.env and add your GOOGLE_API_KEY
```

### Step 3 — Start the Backend

```bash
uvicorn backend.main:app --reload
# API runs at http://localhost:8000
# Swagger docs at http://localhost:8000/docs
```

### Step 4 — Open the Frontend

Simply open `docs/index.html` in your browser — no build step needed.

> **Tip:** For camera capture to work, serve via a local server:
> ```bash
> cd docs && python -m http.server 3000
> # Then open http://localhost:3000
> ```

---

## 🏗️ Project Structure

```
TheDOMinators2026/
├── docs/
│   ├── index.html          ← Single-page app
│   ├── style.css           ← Dark-mode premium UI
│   └── main.js             ← Drag-drop, camera, API calls, results
├── backend/
│   ├── main.py             ← FastAPI server (POST /predict, POST /gemini-analyze)
│   ├── utils.py            ← Preprocessing, condition metadata, Gemini prompt
│   ├── model/
│   │   └── dermascan.keras ← Trained model (from Colab — not committed)
│   └── .env.example        ← API key template
├── colab/
│   └── DermaScan_Training.ipynb  ← Full training notebook
├── requirements.txt
└── README.md
```

---

## 🔌 API Reference

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/health` | Server + model status |
| `POST` | `/predict` | CNN skin classification |
| `POST` | `/gemini-analyze` | Gemini 2.0 Flash report |

Full interactive docs at `http://localhost:8000/docs`

---

## ⚠️ Medical Disclaimer

DermaScan is an **experimental AI tool** built for educational and research purposes. It is **not a medical device** and does not provide medical diagnoses. Always consult a qualified dermatologist or healthcare professional for any skin concerns.

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| ML Model | TensorFlow / Keras, EfficientNetB0 |
| Dataset | HAM10000 (ISIC Archive, Kaggle) |
| AI Report | Google Gemini 2.0 Flash (`google-genai`) |
| Backend | FastAPI + Uvicorn |
| Frontend | Vanilla HTML / CSS / JS |
| Training | Google Colab (T4 GPU) |
