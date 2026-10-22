#!/usr/bin/env python3

# CPU-only image captioning app.
# run with: streamlit run app.py

import logging

import streamlit as st
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_NEW_TOKENS = 30
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]

# Setup the logging mechanism.
log = logging.getLogger("caption_app")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
if not log.handlers:  # streamlit reruns the script, avoid duplicate handlers.
    log.addHandler(ch)


# load the model only once per process, streamlit reruns the script on every interaction.
@st.cache_resource(show_spinner="Loading captioning model...")
def load_model():
    log.info(f"loading {MODEL_NAME} on cpu")
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()
    return processor, model


def generate_caption(image: Image.Image, processor, model) -> str:
    inputs = processor(images=image.convert("RGB"), return_tensors="pt")
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    return processor.decode(output[0], skip_special_tokens=True)


st.set_page_config(page_title="Image Captioning")
st.title("CPU Image Captioning")
st.caption(f"Model: `{MODEL_NAME}` running on CPU")

processor, model = load_model()

uploaded = st.file_uploader("Upload an image", type=IMAGE_TYPES)
if uploaded is not None:
    try:
        image = Image.open(uploaded)
        image.load()  # force decode now so corrupt files fail here.
    except Exception as e:
        log.warning(f"could not read {uploaded.name}: {e}")
        st.error(f"Could not read `{uploaded.name}` as an image.")
        st.stop()

    st.image(image, caption=uploaded.name, width="stretch")

    if st.button("Generate caption", type="primary"):
        log.info(f"captioning {uploaded.name} ({uploaded.size} bytes)")
        with st.spinner("Generating caption..."):
            caption = generate_caption(image, processor, model)
        st.success(caption)
