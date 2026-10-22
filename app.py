#!/usr/bin/env python3

# CPU-only image captioning app.
# run with: streamlit run app.py

import io
import logging
import time
import zipfile
from pathlib import PurePosixPath

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from transformers import BlipForConditionalGeneration, BlipProcessor

MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_NEW_TOKENS = 30
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
MAX_BULK_IMAGES = 50  # cap per ZIP so a single upload can't hog the CPU for too long.
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # skip ZIP entries larger than 10 MB (uncompressed).

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


def open_image(source) -> Image.Image:
    image = Image.open(source)
    image.load()  # force decode now so corrupt files fail here.
    return image


def list_zip_images(archive: zipfile.ZipFile):
    """Split ZIP entries into images to caption and skipped entries with a reason."""
    images, skipped = [], []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or path.parts[0] == "__MACOSX" or path.name.startswith("."):
            continue  # folders and OS metadata, not user files.
        if path.suffix.lower().lstrip(".") not in IMAGE_TYPES:
            skipped.append({"file": info.filename, "reason": "not an image"})
        elif info.file_size > MAX_IMAGE_BYTES:
            skipped.append({"file": info.filename, "reason": "too large"})
        elif len(images) >= MAX_BULK_IMAGES:
            skipped.append({"file": info.filename, "reason": f"over the {MAX_BULK_IMAGES} image limit"})
        else:
            images.append(info)
    return images, skipped


def caption_zip(uploaded, processor, model):
    """Caption every image in the ZIP, one at a time. Returns (results, skipped) or None for a bad ZIP."""
    try:
        archive = zipfile.ZipFile(uploaded)
    except zipfile.BadZipFile:
        log.warning(f"{uploaded.name} is not a valid zip file")
        st.error(f"`{uploaded.name}` is not a valid ZIP file.")
        return None

    results = []
    with archive:
        images, skipped = list_zip_images(archive)
        progress = st.progress(0.0)
        for i, info in enumerate(images):
            progress.progress(i / len(images), text=f"Captioning {info.filename} ({i + 1}/{len(images)})")

            # a bad entry is skipped, it must not stop the rest of the batch.
            try:
                with archive.open(info) as f:
                    image = open_image(io.BytesIO(f.read()))
            except Exception as e:
                log.warning(f"could not read {info.filename}: {e}")
                skipped.append({"file": info.filename, "reason": "corrupt or unreadable"})
                continue

            try:
                start = time.time()
                caption = generate_caption(image, processor, model)
                time_taken = time.time() - start
            except Exception as e:
                log.error(f"captioning failed for {info.filename}: {e}")
                skipped.append({"file": info.filename, "reason": "captioning failed"})
                continue

            results.append({"file": info.filename, "caption": caption, "seconds": round(time_taken, 2)})
        progress.empty()

    return results, skipped


def single_mode(processor, model):
    uploaded = st.file_uploader("Upload an image", type=IMAGE_TYPES, key="single_upload")
    if uploaded is None:
        return

    try:
        image = open_image(uploaded)
    except Exception as e:
        log.warning(f"could not read {uploaded.name}: {e}")
        st.error(f"Could not read `{uploaded.name}` as an image.")
        return

    st.image(image, caption=uploaded.name, width="stretch")

    if st.button("Generate caption", type="primary", key="single_button"):
        log.info(f"captioning {uploaded.name} ({uploaded.size} bytes)")
        try:
            with st.spinner("Generating caption..."):
                caption = generate_caption(image, processor, model)
        except Exception as e:
            log.error(f"captioning failed for {uploaded.name}: {e}")
            st.error("Captioning failed for this image.")
            return
        st.success(caption)


def bulk_mode(processor, model):
    st.caption(f"Up to {MAX_BULK_IMAGES} images ({', '.join(IMAGE_TYPES)}) per ZIP, "
               f"{MAX_IMAGE_BYTES // (1024 * 1024)} MB max per image. Other files are skipped.")
    uploaded = st.file_uploader("Upload a ZIP of images", type=["zip"], key="bulk_upload")
    if uploaded is None:
        return

    if st.button("Caption all images", type="primary", key="bulk_button"):
        log.info(f"bulk captioning {uploaded.name} ({uploaded.size} bytes)")
        start = time.time()
        outcome = caption_zip(uploaded, processor, model)
        if outcome is None:
            return
        results, skipped = outcome
        log.info(f"bulk done: {len(results)} captioned, {len(skipped)} skipped in {time.time() - start:.2f}s")
        # keep the results across reruns (e.g. switching tabs), until a different ZIP is uploaded.
        st.session_state.bulk = {"file_id": uploaded.file_id, "results": results,
                                 "skipped": skipped, "seconds": time.time() - start}

    bulk = st.session_state.get("bulk")
    if bulk is None or bulk["file_id"] != uploaded.file_id:
        return

    results, skipped = bulk["results"], bulk["skipped"]
    if results:
        st.success(f"Captioned {len(results)} image(s) in {bulk['seconds']:.1f}s, skipped {len(skipped)}.")
        df = pd.DataFrame(results)
        st.dataframe(df, hide_index=True, width="stretch")
        st.download_button("Download captions (CSV)", df.to_csv(index=False), file_name="captions.csv",
                           mime="text/csv", on_click="ignore")
    else:
        st.warning("No images could be captioned from this ZIP.")

    if skipped:
        with st.expander(f"Skipped files ({len(skipped)})"):
            st.dataframe(pd.DataFrame(skipped), hide_index=True, width="stretch")


st.set_page_config(page_title="Image Captioning")
st.title("CPU Image Captioning")
st.caption(f"Model: `{MODEL_NAME}` running on CPU")

processor, model = load_model()

single_tab, bulk_tab = st.tabs(["Single image", "Bulk (ZIP)"])
with single_tab:
    single_mode(processor, model)
with bulk_tab:
    bulk_mode(processor, model)
