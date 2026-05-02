"""
DermaScan — FastAPI Backend
===========================
Endpoints:
  GET  /health           Health check
  POST /predict          CNN skin-condition classification (EfficientNetB0)
  POST /gemini-analyze   Gemini 2.0 Flash multimodal dermatology report
"""

import json
import logging
import os
from contextlib import asynccontextmanager

import numpy as np
from dotenv import load_dotenv
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware

from utils import build_gemini_prompt, postprocess_prediction, preprocess_image

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")
logger = logging.getLogger("dermascan")

# ─── Global model handle ──────────────────────────────────────────────────────
_model = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the Keras model once at startup."""
    global _model
    model_path = os.path.join(os.path.dirname(__file__), "model", "dermascan.keras")
    if os.path.exists(model_path):
        try:
            import tensorflow as tf
            _model = tf.keras.models.load_model(model_path)
            logger.info(f"✅  Model loaded from {model_path}")
        except Exception as exc:
            logger.warning(f"⚠️  Model load failed: {exc}")
    else:
        logger.warning(
            f"⚠️  No model at '{model_path}'. "
            "Run colab/DermaScan_Training.ipynb and copy dermascan.keras here."
        )
    yield


# ─── App setup ────────────────────────────────────────────────────────────────
app = FastAPI(
    title="DermaScan API",
    description="AI-powered skin condition analysis (CNN + Gemini 2.0 Flash)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["system"])
async def health():
    return {
        "status": "ok",
        "model_loaded": _model is not None,
        "demo_mode": _model is None,
    }


@app.post("/predict", tags=["ml"])
async def predict(file: UploadFile = File(...)):
    """
    Run EfficientNetB0 CNN classification on the uploaded skin image.
    Returns the top predicted condition plus top-3 ranked alternatives.
    If the model is not loaded, returns realistic demo data.
    """
    img_bytes = await file.read()

    if _model is None:
        # Demo mode — return plausible mock data when no model is loaded
        logger.info("Demo mode: returning mock CNN prediction.")
        return _demo_predict()

    img_array = preprocess_image(img_bytes)
    raw_probs = _model.predict(img_array, verbose=0)[0]
    return postprocess_prediction(raw_probs)


@app.post("/gemini-analyze", tags=["gemini"])
async def gemini_analyze(
    file: UploadFile = File(...),
    cnn_result: str = Form(default=""),
):
    """
    Send the skin image to Gemini 2.0 Flash for a rich, natural-language
    dermatology report including care tips and urgency assessment.
    """
    api_key = os.getenv("GOOGLE_API_KEY", "")
    if not api_key or api_key == "your_gemini_api_key_here":
        logger.warning("GOOGLE_API_KEY not set — returning demo Gemini report.")
        return _demo_gemini()

    img_bytes = await file.read()
    content_type = file.content_type or "image/jpeg"

    # Parse CNN context if provided
    cnn_data: dict = {}
    if cnn_result:
        try:
            cnn_data = json.loads(cnn_result)
        except json.JSONDecodeError:
            pass

    prompt = build_gemini_prompt(cnn_data)

    try:
        from google import genai
        from google.genai import types

        client = genai.Client(api_key=api_key)
        response = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=[
                types.Part.from_bytes(data=img_bytes, mime_type=content_type),
                prompt,
            ],
        )
        return _parse_gemini_response(response.text)
    except Exception as exc:
        logger.error(f"Gemini API error: {exc}")
        return {"error": str(exc), "fallback": _demo_gemini()}


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _parse_gemini_response(text: str) -> dict:
    """Extract JSON from Gemini's response (handles markdown code fences)."""
    clean = text.strip()
    if "```json" in clean:
        clean = clean.split("```json")[1].split("```")[0].strip()
    elif "```" in clean:
        clean = clean.split("```")[1].split("```")[0].strip()
    try:
        return json.loads(clean)
    except json.JSONDecodeError:
        return {"raw_text": text, "parse_error": True}


def _demo_predict() -> dict:
    """Realistic mock CNN response for demo / no-model mode."""
    return {
        "condition": "Melanocytic Nevi",
        "common_name": "Common Mole",
        "code": "nv",
        "confidence": 0.8712,
        "severity": "low",
        "severity_label": "Low Risk",
        "seek_doctor": False,
        "description": (
            "Melanocytic nevi are common benign moles formed by clusters of "
            "pigment-producing melanocytes. They are typically harmless but "
            "should be monitored for changes in size, shape, or color."
        ),
        "color": "emerald",
        "top3": [
            {"code": "nv",  "label": "Melanocytic Nevi",  "probability": 0.8712},
            {"code": "bkl", "label": "Benign Keratosis",  "probability": 0.0831},
            {"code": "df",  "label": "Dermatofibroma",    "probability": 0.0314},
        ],
        "_demo": True,
    }


def _demo_gemini() -> dict:
    """Realistic mock Gemini report for demo / no-key mode."""
    return {
        "visual_observations": (
            "The image shows a well-defined, uniformly pigmented lesion with "
            "smooth, regular borders. The coloration appears consistent throughout "
            "with no visible asymmetry or multi-tonal variation."
        ),
        "likely_condition": "Melanocytic Nevi (Common Mole)",
        "explanation": (
            "This appears to be a common benign mole — a harmless cluster of "
            "melanocytes that form a pigmented spot on the skin. Such lesions are "
            "extremely common and typically do not require treatment."
        ),
        "urgency": "monitor",
        "urgency_label": "Keep an eye on it",
        "urgency_reason": "Lesion appears benign, but routine monitoring is advisable.",
        "care_tips": [
            "Apply broad-spectrum SPF 30+ sunscreen daily to prevent further pigmentation changes.",
            "Photograph the area monthly to track any changes in size, shape, or color.",
            "Avoid picking or scratching the lesion to prevent irritation or infection.",
        ],
        "doctor_questions": [
            "Does this mole show any signs of the ABCDE criteria (Asymmetry, Border, Color, Diameter, Evolution)?",
            "Should I schedule a full-body skin check given my sun exposure history?",
            "At what point would you recommend a biopsy for this type of lesion?",
        ],
        "disclaimer": (
            "This AI analysis is for informational purposes only and does not "
            "constitute medical advice. Always consult a qualified healthcare "
            "professional for diagnosis and treatment."
        ),
        "_demo": True,
    }
