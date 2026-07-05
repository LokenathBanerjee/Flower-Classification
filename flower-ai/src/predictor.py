

import numpy as np
from PIL import Image, UnidentifiedImageError

from src.utils import load_json_file

IMG_SIZE = (224, 224)

CONFIDENCE_THRESHOLD = 0.50


class FlowerPredictor:
    """Wraps the trained EfficientNetB0 model for inference."""

    def __init__(self, model_path: str, class_names_path: str):
        self.class_names = load_json_file(class_names_path)
        if not isinstance(self.class_names, list) or len(self.class_names) == 0:
            raise ValueError(
                f"'{class_names_path}' should contain a non-empty JSON list of class names."
            )

        self.model = self._load_model(model_path)
        self._validate_output_shape()

    def _load_model(self, model_path: str):
        import os

        if not os.path.exists(model_path):
            raise FileNotFoundError(
                f"Model file not found at '{model_path}'. "
                f"Train the model in Colab, download the .keras file, and place it there."
            )

        import tensorflow as tf

        try:
            return tf.keras.models.load_model(model_path)
        except Exception as e:

            raise RuntimeError(
                f"Could not load the model at '{model_path}'. This usually means a "
                f"TensorFlow/Keras version mismatch between Colab (training) and this "
                f"environment (inference). Original error: {e}"
            )

    def _validate_output_shape(self):
 
        output_units = self.model.output_shape[-1]
        if output_units != len(self.class_names):
            raise ValueError(
                f"Model outputs {output_units} classes but class_names.json has "
                f"{len(self.class_names)} entries. These must match exactly, "
                f"and the order must match the order used during training."
            )

    def preprocess(self, image: Image.Image) -> np.ndarray:

        image = image.convert("RGB")
        image = image.resize(IMG_SIZE, Image.LANCZOS)
        array = np.asarray(image, dtype=np.float32)   # raw 0-255, no rescaling
        return np.expand_dims(array, axis=0)

    def predict(self, image: Image.Image, top_k: int = 3) -> dict:

        if image is None:
            raise ValueError("No image was provided to predict().")

        try:
            batch = self.preprocess(image)
        except (UnidentifiedImageError, OSError) as e:
            raise ValueError(f"This file doesn't look like a readable image: {e}")

        raw_scores = self.model.predict(batch, verbose=0)[0]

        ranked_indices = np.argsort(raw_scores)[::-1][:top_k]
        top_predictions = [(self.class_names[i], float(raw_scores[i])) for i in ranked_indices]

        best_name, best_confidence = top_predictions[0]
        return {
            "flower": best_name,
            "confidence": best_confidence,
            "is_confident": best_confidence >= CONFIDENCE_THRESHOLD,
            "top_predictions": top_predictions,
        }
