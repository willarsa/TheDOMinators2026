"""DermaScan — Image preprocessing utilities and condition metadata."""

import io
import numpy as np
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE = (100, 75)  # Width, Height (matches training [75, 100])

# HAM10000 classes sorted alphabetically (matches label encoder order after fit)
CLASS_NAMES = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]

# ─── Condition Metadata ───────────────────────────────────────────────────────

CONDITION_METADATA = {
    "akiec": {
        "label": "Actinic Keratoses",
        "common_name": "Actinic Keratosis",
        "severity": "medium",
        "severity_label": "Moderate Risk",
        "seek_doctor": True,
        "description": (
            "Actinic keratoses are rough, scaly patches on the skin caused by years "
            "of sun exposure. They are considered precancerous and can progress to "
            "squamous cell carcinoma if left untreated."
        ),
        "color": "amber",
    },
    "bcc": {
        "label": "Basal Cell Carcinoma",
        "common_name": "Basal Cell Carcinoma",
        "severity": "high",
        "severity_label": "High Risk",
        "seek_doctor": True,
        "description": (
            "Basal cell carcinoma is the most common type of skin cancer. While it "
            "rarely spreads to other parts of the body, it requires prompt medical "
            "treatment to prevent local tissue damage."
        ),
        "color": "red",
    },
    "bkl": {
        "label": "Benign Keratosis",
        "common_name": "Benign Keratosis",
        "severity": "low",
        "severity_label": "Low Risk",
        "seek_doctor": False,
        "description": (
            "Benign keratosis includes seborrheic keratoses and solar lentigines — "
            "harmless, non-cancerous skin growths that become more common with age. "
            "They typically do not require treatment unless cosmetically bothersome."
        ),
        "color": "emerald",
    },
    "df": {
        "label": "Dermatofibroma",
        "common_name": "Dermatofibroma",
        "severity": "low",
        "severity_label": "Low Risk",
        "seek_doctor": False,
        "description": (
            "Dermatofibromas are firm, harmless bumps that commonly appear on the "
            "legs. They are benign fibrous nodules and rarely require treatment "
            "unless they cause discomfort."
        ),
        "color": "emerald",
    },
    "mel": {
        "label": "Melanoma",
        "common_name": "Melanoma",
        "severity": "high",
        "severity_label": "High Risk",
        "seek_doctor": True,
        "description": (
            "Melanoma is the most serious form of skin cancer, developing in the "
            "cells that give skin its color. Early detection is critical — please "
            "seek professional medical evaluation immediately."
        ),
        "color": "red",
    },
    "nv": {
        "label": "Melanocytic Nevi",
        "common_name": "Common Mole",
        "severity": "low",
        "severity_label": "Low Risk",
        "seek_doctor": False,
        "description": (
            "Melanocytic nevi are common benign moles formed by clusters of "
            "pigment-producing melanocytes. They are typically harmless but should "
            "be monitored for changes in size, shape, or color."
        ),
        "color": "emerald",
    },
    "vasc": {
        "label": "Vascular Lesions",
        "common_name": "Vascular Lesion",
        "severity": "low",
        "severity_label": "Low Risk",
        "seek_doctor": False,
        "description": (
            "Vascular lesions include angiomas, angiokeratomas, and pyogenic "
            "granulomas — benign growths of blood vessels in the skin. They are "
            "usually harmless and often purely cosmetic in nature."
        ),
        "color": "emerald",
    },
}


# ─── Image Preprocessing ──────────────────────────────────────────────────────

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes for EfficientNetB0 inference.
    Returns float32 array of shape (1, 224, 224, 3).
    EfficientNetB0 expects pixel values in [-1, 1].
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    arr = arr / 255.0  # Standard [0, 1] scaling
    return np.expand_dims(arr, axis=0)  # (1, 75, 100, 3)


# ─── Postprocessing ───────────────────────────────────────────────────────────

def postprocess_prediction(raw_probs: np.ndarray, top_n: int = 3) -> dict:
    """
    Convert raw softmax probabilities into a structured prediction dict.

    Args:
        raw_probs: 1-D float array of length 7 (one per class)
        top_n:     number of top predictions to include

    Returns:
        dict with condition metadata, confidence, severity, and top-N list
    """
    class_idx = int(np.argmax(raw_probs))
    code = CLASS_NAMES[class_idx]
    confidence = float(raw_probs[class_idx])
    meta = CONDITION_METADATA[code]

    top_indices = np.argsort(raw_probs)[::-1][:top_n]
    top_predictions = [
        {
            "code": CLASS_NAMES[i],
            "label": CONDITION_METADATA[CLASS_NAMES[i]]["label"],
            "probability": round(float(raw_probs[i]), 4),
        }
        for i in top_indices
    ]

    return {
        "condition": meta["label"],
        "common_name": meta["common_name"],
        "code": code,
        "confidence": round(confidence, 4),
        "severity": meta["severity"],
        "severity_label": meta["severity_label"],
        "seek_doctor": meta["seek_doctor"],
        "description": meta["description"],
        "color": meta["color"],
        "top3": top_predictions,
    }


# ─── Gemini Prompt Builder ────────────────────────────────────────────────────

def build_gemini_prompt(cnn_data: dict, age: str = "", gender: str = "", localization: str = "") -> str:
    """Build a structured prompt for Gemini analysis with patient context."""
    patient_context = ""
    if age or gender or localization:
        parts = []
        if age: parts.append(f"Age: {age}")
        if gender: parts.append(f"Sex: {gender}")
        if localization: parts.append(f"Location: {localization}")
        patient_context = f"\n\nPatient Details: {', '.join(parts)}"

    cnn_hint = ""
    if cnn_data.get("condition"):
        prob_pct = round(cnn_data.get("confidence", 0) * 100)
        cnn_hint = (
            f"\n\nContext: Our specialized dermatology CNN classified the patient's skin condition as "
            f"'{cnn_data['condition']}' with {prob_pct}% confidence. "
            f"The underlying model prediction code is '{cnn_data.get('code', 'unknown')}'. "
            f"The severity is considered '{cnn_data.get('severity_label', 'Unknown')}'. "
            f"Model description: {cnn_data.get('description', '')}"
        )

    return f"""You are an expert dermatology AI assistant. Provide a patient-friendly explanation based ONLY on the CNN diagnosis and patient context provided below. Respond with ONLY a valid JSON object — no markdown, no preamble, no trailing text.{patient_context}{cnn_hint}

Return exactly this JSON structure:
{{
  "visual_observations": "Provide a brief medical description of what this condition typically looks like (2-3 sentences)",
  "likely_condition": "Skin condition name based on the context",
  "explanation": "Plain-English explanation of what this condition is (2-3 sentences)",
  "urgency": "monitor",
  "urgency_label": "Keep an eye on it",
  "urgency_reason": "Brief reason for this urgency level",
  "care_tips": ["Tip one", "Tip two", "Tip three"],
  "doctor_questions": ["Question one", "Question two", "Question three"],
  "disclaimer": "This AI analysis is for informational purposes only and does not constitute medical advice. Always consult a qualified healthcare professional."
}}

For the urgency field use ONLY: "monitor", "schedule", or "seek_care"
CRITICAL: Return ONLY the JSON object. Nothing else."""
