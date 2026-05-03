import os
import tensorflow as tf
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("test_model")

model_path = os.path.join("backend", "model", "mylesion.keras")
if os.path.exists(model_path):
    try:
        model = tf.keras.models.load_model(model_path)
        logger.info(f"✅ Model loaded successfully from {model_path}")
        print("SUCCESS")
    except Exception as e:
        logger.error(f"❌ Model load failed: {e}")
        print(f"FAILURE: {e}")
else:
    logger.error(f"❌ Model path not found: {model_path}")
    print("MISSING")
