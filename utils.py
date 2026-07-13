import cv2
import numpy as np
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

EMO_LABELS = ["angry","disgust","fear","happy","sad","surprise","neutral"]

def ensure_vader():
    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        nltk.download("vader_lexicon")

# Load OpenCV Haar cascade once
_CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def _detect_face(gray):
    faces = _CASCADE.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(60, 60),
        flags=cv2.CASCADE_SCALE_IMAGE
    )
    if len(faces) == 0:
        return None
    # largest face
    x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
    return x, y, w, h

def preprocess_face(frame_bgr):
    """
    Return 48x48 grayscale normalized array shaped (1,1,48,48).
    Tries Haar face; falls back to centered square.
    """
    if frame_bgr is None or frame_bgr.size == 0:
        return np.zeros((1, 1, 48, 48), dtype=np.float32)

    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    box = _detect_face(gray)
    if box is not None:
        x, y, w, h = box
        pad = int(0.15 * max(w, h))  # small margin
        x0 = max(0, x - pad); y0 = max(0, y - pad)
        x1 = min(gray.shape[1], x + w + pad); y1 = min(gray.shape[0], y + h + pad)
        crop = gray[y0:y1, x0:x1]
    else:
        h, w = gray.shape
        size = min(h, w)
        y0 = (h - size) // 2
        x0 = (w - size) // 2
        crop = gray[y0:y0+size, x0:x0+size]

    crop = cv2.resize(crop, (48, 48), interpolation=cv2.INTER_AREA)
    crop = crop.astype(np.float32) / 255.0
    crop = (crop - 0.5) / 0.5  # normalize
    return crop[None, None, :, :]

def vader_score(text: str) -> float:
    ensure_vader()
    sid = SentimentIntensityAnalyzer()
    return sid.polarity_scores(text).get("compound", 0.0)
