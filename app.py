import streamlit as st
import torch
import json
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import os

st.set_page_config(page_title="EduQuery Classifier", layout="centered")


BASE_DIR = os.path.dirname(__file__)
MODEL_PATH = os.path.join(BASE_DIR, "src", "models")

# Load model
@st.cache_resource
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)

    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_PATH,
        low_cpu_mem_usage=True
    )

    model.eval()

    with open(os.path.join(MODEL_PATH, "label_map.json")) as f:
        id2label = json.load(f)

    return tokenizer, model, id2label

tokenizer, model, id2label = load_model()

# UI
st.title("🎓 Educational Query Intent Classifier")
st.markdown("Classify student queries into learning intents.")

text = st.text_area("Enter your query:")


@torch.no_grad()
def predict(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    pred = torch.argmax(probs).item()
    confidence = probs[0][pred].item()
    return id2label[str(pred)], confidence

if st.button("Classify"):
    if text.strip():
        label, conf = predict(text)

        st.subheader("Prediction:")
        st.write(f"**{label}** ({conf:.2f} confidence)")

        # Smart routing suggestion
        st.subheader("Suggested Action:")
        if label == "CONCEPTUAL":
            st.info("Show explanation resources")
        elif label == "NAVIGATIONAL":
            st.info("Redirect to specific page")
        elif label == "PROCEDURAL":
            st.info("Provide step-by-step solution")
        elif label == "ADVANCED":
            st.info("Show deeper technical material")

    else:
        st.warning("Please enter a query.")
