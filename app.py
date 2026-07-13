
import os
import time
import cv2
import torch
import numpy as np
import gradio as gr
from typing import Tuple
from model import MiniEmotionCNN
from utils import preprocess_face, vader_score, EMO_LABELS

DEVICE = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
MODEL_PATH = os.getenv("EMOTION_MODEL_PATH", "weights/emotion_cnn.pt")

def load_model() -> MiniEmotionCNN:
    model = MiniEmotionCNN(num_classes=len(EMO_LABELS))
    if os.path.exists(MODEL_PATH):
        state = torch.load(MODEL_PATH, map_location="cpu")
        model.load_state_dict(state)
    model.eval()
    model.to(DEVICE)
    return model

MODEL = load_model()

def fuse(face_logits: np.ndarray, text_compound: float) -> Tuple[str, dict]:
    """
    Combine face logits (softmax) and VADER compound score to pick a final label.
    Simple heuristic: bias toward happy for high compound, toward sad/angry for low compound.
    """
    probs = torch.softmax(torch.tensor(face_logits), dim=-1).numpy()
    bias = np.zeros_like(probs)
    if text_compound > 0.35:
        happy_idx = EMO_LABELS.index("happy")
        bias[happy_idx] += text_compound * 0.4
    elif text_compound < -0.35:
        sad_idx = EMO_LABELS.index("sad")
        angry_idx = EMO_LABELS.index("angry")
        bias[sad_idx] += (-text_compound) * 0.3
        bias[angry_idx] += (-text_compound) * 0.2
    fused = probs + bias
    fused = fused / fused.sum()
    top_idx = int(np.argmax(fused))
    return EMO_LABELS[top_idx], {EMO_LABELS[i]: float(fused[i]) for i in range(len(EMO_LABELS))}

def empathetic_reply(label: str, text_compound: float) -> str:
    templates = {
        "happy": "You look cheerful! 🎉 I’m glad to hear that. Want to share what made your day better?",
        "surprise": "That looks surprising! 😮 Did something unexpected happen?",
        "neutral": "I’m here with you. Tell me a bit about what’s on your mind.",
        "sad": "I’m sorry you’re feeling down. 💙 Want to talk through it together? I’m listening.",
        "angry": "I can sense some frustration. 😤 Do you want to vent about it?",
        "fear": "It sounds like there’s some worry. 🫶 What would help you feel safer right now?",
        "disgust": "That sounds upsetting. Want to tell me what felt wrong about it?",
    }
    # light tweak with text sentiment
    if text_compound > 0.5 and label != "happy":
        return "I’m picking up positive words — that’s awesome! Tell me more about the good news. 🌟"
    if text_compound < -0.5 and label == "neutral":
        return "I sense some heaviness in your words. I'm here. What happened?"
    return templates.get(label, "I’m here with you. Tell me more.")

def predict_from_frame(frame_bgr: np.ndarray, text: str):
    # Preprocess and run model
    x = preprocess_face(frame_bgr)
    with torch.no_grad():
        xt = torch.from_numpy(x).to(DEVICE)
        logits = MODEL(xt).cpu().numpy()[0]
    txt_score = vader_score(text or "")
    label, dist = fuse(logits, txt_score)
    reply = empathetic_reply(label, txt_score)
    return label, dist, reply

def inference(image: np.ndarray, text: str):
    if image is None:
        # fallback blank image to avoid crashes
        image = np.zeros((480, 640, 3), dtype=np.uint8)
    label, dist, reply = predict_from_frame(image, text or "")
    # return label, JSON-like dict for Plot, and reply
    return label, dist, reply

with gr.Blocks(title="Emotion-Aware Assistant") as demo:
    gr.Markdown("# Emotion-Aware Virtual Assistant (Face + Text)")
    with gr.Row():
        image = gr.Image(label="Webcam / Image", sources=["webcam", "upload"], streaming=False)
        text = gr.Textbox(label="Your message (optional)", placeholder="Type how you feel...")
    with gr.Row():
        out_label = gr.Textbox(label="Detected Emotion")
        out_probs = gr.Label(label="Probabilities")
    out_reply = gr.Textbox(label="Assistant Reply", lines=2)
    btn = gr.Button("Analyze")
    btn.click(fn=inference, inputs=[image, text], outputs=[out_label, out_probs, out_reply])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860, show_error=True, debug=False)
