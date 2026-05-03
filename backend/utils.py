"""MyLesion — Image preprocessing utilities and condition metadata."""

import io
import numpy as np
from PIL import Image

# ─── Constants ────────────────────────────────────────────────────────────────

IMG_SIZE = (224, 224)  # Width, Height (matches EfficientNetB0 training)

# SCIN classes sorted alphabetically
CLASS_NAMES = [
    "Acne",
    "Eczema",
    "Folliculitis",
    "Healthy Skin",
    "Herpes Simplex",
    "Impetigo",
    "Insect Bite",
    "Psoriasis",
    "Rosacea",
    "Tinea",
    "Urticaria"
]

# ─── Condition Metadata ───────────────────────────────────────────────────────

CONDITION_METADATA = {
    "Acne": {
        "label": "Acne",
        "common_name": "Pimples / Acne",
        "severity": "low",
        "severity_label": "Low Risk",
        "seek_doctor": False,
        "description": (
            "Acne is a common skin condition where pores become clogged with oil and "
            "dead skin cells. It usually presents as blackheads, whiteheads, or pimples."
        ),
        "color": "emerald",
    },
    "Eczema": {
        "label": "Eczema",
        "common_name": "Atopic Dermatitis",
        "severity": "medium",
        "severity_label": "Moderate Risk",
        "seek_doctor": False,
        "description": (
            "Eczema is a condition that makes your skin red and itchy. It's common "
            "in children but can occur at any age. It's long-lasting and tends to flare up."
        ),
        "color": "amber",
    },
    "Folliculitis": {
        "label": "Folliculitis",
        "common_name": "Infected Hair Follicle",
        "severity": "low",
        "severity_label": "Low Risk",
        "seek_doctor": False,
        "description": (
            "Folliculitis is a common skin condition in which hair follicles become "
            "inflamed. It's usually caused by a bacterial or fungal infection."
        ),
        "color": "emerald",
    },
    "Healthy Skin": {
        "label": "Healthy Skin",
        "common_name": "Typical Skin",
        "severity": "low",
        "severity_label": "No Issue Detected",
        "seek_doctor": False,
        "description": (
            "The analyzed area appears to be typical, healthy skin with no discernible "
            "pathology detected by the AI model."
        ),
        "color": "emerald",
    },
    "Herpes Simplex": {
        "label": "Herpes Simplex",
        "common_name": "Cold Sores / Fever Blisters",
        "severity": "medium",
        "severity_label": "Moderate Risk",
        "seek_doctor": True,
        "description": (
            "Herpes simplex is a viral infection that causes sores. It most commonly "
            "appears as cold sores around the mouth or as genital herpes."
        ),
        "color": "amber",
    },
    "Impetigo": {
        "label": "Impetigo",
        "common_name": "School Sores",
        "severity": "high",
        "severity_label": "High Risk (Contagious)",
        "seek_doctor": True,
        "description": (
            "Impetigo is a highly contagious skin infection that mainly affects infants "
            "and children. It usually appears as red sores on the face."
        ),
        "color": "red",
    },
    "Insect Bite": {
        "label": "Insect Bite",
        "common_name": "Bug Bite",
        "severity": "low",
        "severity_label": "Low Risk",
        "seek_doctor": False,
        "description": (
            "Most insect bites and stings are minor and can be treated at home. "
            "They usually cause a small, itchy red bump."
        ),
        "color": "emerald",
    },
    "Psoriasis": {
        "label": "Psoriasis",
        "common_name": "Psoriasis",
        "severity": "medium",
        "severity_label": "Moderate Risk",
        "seek_doctor": True,
        "description": (
            "Psoriasis is a skin disease that causes a rash with itchy, scaly patches, "
            "most commonly on the knees, elbows, trunk and scalp."
        ),
        "color": "amber",
    },
    "Rosacea": {
        "label": "Rosacea",
        "common_name": "Rosacea",
        "severity": "low",
        "severity_label": "Low Risk",
        "seek_doctor": False,
        "description": (
            "Rosacea is a common skin condition that causes blushing or flushing and "
            "visible blood vessels in your face. It may also produce small, pus-filled bumps."
        ),
        "color": "emerald",
    },
    "Tinea": {
        "label": "Tinea",
        "common_name": "Fungal Infection / Ringworm",
        "severity": "medium",
        "severity_label": "Moderate Risk",
        "seek_doctor": True,
        "description": (
            "Tinea is the name of a group of diseases caused by a fungus. Types "
            "include ringworm, athlete's foot and jock itch."
        ),
        "color": "amber",
    },
    "Urticaria": {
        "label": "Urticaria",
        "common_name": "Hives",
        "severity": "medium",
        "severity_label": "Moderate Risk",
        "seek_doctor": False,
        "description": (
            "Hives, also known as urticaria, are itchy, raised welts that are found "
            "on the skin. They are usually red, pink, or flesh-colored."
        ),
        "color": "amber",
    },
}


# ─── Image Preprocessing ──────────────────────────────────────────────────────

def preprocess_image(file_bytes: bytes) -> np.ndarray:
    """
    Preprocess raw image bytes for MobileNetV2 inference.
    Returns float32 array of shape (1, 224, 224, 3).
    MobileNetV2 expects pixel values in [-1, 1].
    """
    img = Image.open(io.BytesIO(file_bytes)).convert("RGB")
    img = img.resize(IMG_SIZE, Image.LANCZOS)
    arr = np.array(img, dtype=np.float32)
    # MobileNetV2 scaling: [0, 255] -> [-1, 1]
    # Matches Rescaling(1./127.5, offset=-1) from training notebook
    arr = (arr / 127.5) - 1.0
    return np.expand_dims(arr, axis=0)  # (1, 224, 224, 3)


# ─── Postprocessing ───────────────────────────────────────────────────────────

def postprocess_prediction(raw_probs: np.ndarray, top_n: int = 3) -> dict:
    """
    Convert raw softmax probabilities into a structured prediction dict.
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

def build_gemini_prompt(cnn_data: dict) -> str:
    """Build a structured prompt for Gemini analysis."""
    cnn_hint = ""
    if cnn_data.get("condition"):
        prob_pct = round(cnn_data.get("confidence", 0) * 100)
        cnn_hint = (
            f"\n\nContext: The MyLesion AI classified this as "
            f"'{cnn_data['condition']}' with {prob_pct}% confidence. "
            f"The underlying model prediction code is '{cnn_data.get('code', 'unknown')}'. "
            f"The severity is considered '{cnn_data.get('severity_label', 'Unknown')}'. "
            f"Clinical reference: {cnn_data.get('description', '')}"
        )

    return f"""You are an expert dermatology AI assistant. Provide a patient-friendly explanation based ONLY on the AI diagnosis provided below. Respond with ONLY a valid JSON object — no markdown, no preamble, no trailing text.{cnn_hint}

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
