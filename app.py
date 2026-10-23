#!/usr/bin/env python3

# CPU-only image captioning app.
# run with: streamlit run app.py
# prometheus metrics are exposed at http://localhost:18000/metrics

import io
import logging
import time
import zipfile
from pathlib import PurePosixPath

import pandas as pd
import streamlit as st
import torch
from PIL import Image
from prometheus_client import start_http_server
from streamlit.runtime.scriptrunner import get_script_run_ctx
from transformers import BlipForConditionalGeneration, BlipProcessor

import metrics

MODEL_NAME = "Salesforce/blip-image-captioning-base"
MAX_NEW_TOKENS = 30
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp", "bmp"]
MAX_BULK_IMAGES = 50  # cap per ZIP so a single upload can't hog the CPU for too long.
MAX_IMAGE_BYTES = 10 * 1024 * 1024  # skip ZIP entries larger than 10 MB (uncompressed).
METRICS_PORT = 18000
MODES = ("single", "bulk")

# reason code (metric label) -> text shown in the UI.
SKIP_REASONS = {
    "not_image": "not an image",
    "too_large": "too large",
    "over_limit": f"over the {MAX_BULK_IMAGES} image limit",
    "corrupt": "corrupt or unreadable",
    "caption_failed": "captioning failed",
}

# Setup the logging mechanism.
log = logging.getLogger("caption_app")
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
ch.setFormatter(formatter)
if not log.handlers:  # streamlit reruns the script, avoid duplicate handlers.
    log.addHandler(ch)


# start the exporter metrics service once per process.
@st.cache_resource
def start_metrics_server():
    log.info(f"starting the prometheus monitor at port {METRICS_PORT}")
    try:
        start_http_server(METRICS_PORT)
    except OSError as e:
        log.warning(f"metrics server not started, port {METRICS_PORT} in use: {e}")
    # add build information to the info metric.
    metrics.build.info({'version': '0.1.0', 'model': MODEL_NAME, 'device': 'cpu',
                        'torch': torch.__version__, 'author': 'Sujal Kumar', 'builddate': 'October 2026'})

    # export 0 for the label values known up front. a series that first appears with value 1
    # is invisible to rate()/increase() in prometheus, so its first event would be missed.
    for mode in MODES:
        for metric in (metrics.active_requests, metrics.inference_seconds, metrics.image_pixels,
                       metrics.upload_bytes, metrics.request_seconds, metrics.preprocess_seconds,
                       metrics.caption_words):
            metric.labels(mode=mode)
        for error_type in ("invalid_image", "inference_error"):
            metrics.errors.labels(mode=mode, error_type=error_type)
    metrics.errors.labels(mode="bulk", error_type="bad_zip")
    for reason in SKIP_REASONS:
        metrics.zip_entries_skipped.labels(reason=reason)


# load the model only once per process, streamlit reruns the script on every interaction.
@st.cache_resource(show_spinner="Loading captioning model...")
def load_model():
    log.info(f"loading {MODEL_NAME} on cpu")
    start = time.time()
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_NAME)
    model.eval()

    metrics.model_load_seconds.set(time.time() - start)
    metrics.model_size_bytes.set(sum(p.numel() * p.element_size() for p in model.parameters()))
    metrics.model_loaded.labels(model=MODEL_NAME).set(1)
    return processor, model


def browser_name(user_agent: str) -> str:
    # order matters: Edge's user agent also contains Chrome, and Chrome's contains Safari.
    for token, name in (("Edg/", "edge"), ("Firefox/", "firefox"), ("Chrome/", "chrome"), ("Safari/", "safari")):
        if token in user_agent:
            return name
    return "other"


def client_labels() -> dict:
    """Client metadata attached to the per-client metrics."""
    ctx = get_script_run_ctx()
    return {
        "client_ip": st.context.ip_address or "localhost",  # streamlit returns None for localhost.
        "session_id": ctx.session_id[:8] if ctx is not None else "unknown",
    }


def init_session_metrics():
    """Create this session's per-client series at 0 on its first run, for the same reason as in
    start_metrics_server: otherwise the first request of every new session would be missed."""
    if st.session_state.get("metrics_ready"):
        return
    client = client_labels()
    browser = browser_name(st.context.headers.get("User-Agent", ""))
    for mode in MODES:
        metrics.requests.labels(mode=mode, browser=browser, **client)
        for status in ("success", "failed"):
            metrics.images_processed.labels(mode=mode, status=status, **client)
    st.session_state.metrics_ready = True


def record_request(mode: str, uploaded, client: dict):
    browser = browser_name(st.context.headers.get("User-Agent", ""))
    metrics.requests.labels(mode=mode, browser=browser, **client).inc()
    metrics.upload_bytes.labels(mode=mode).observe(uploaded.size)


def generate_caption(image: Image.Image, processor, model, mode: str) -> str:
    metrics.image_pixels.labels(mode=mode).observe(image.width * image.height)

    with metrics.preprocess_seconds.labels(mode=mode).time():
        inputs = processor(images=image.convert("RGB"), return_tensors="pt")

    start = time.time()
    with torch.inference_mode():
        output = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS)
    time_taken = time.time() - start
    metrics.inference_seconds.labels(mode=mode).observe(time_taken)
    metrics.last_inference.labels(mode=mode).set(time_taken)

    caption = processor.decode(output[0], skip_special_tokens=True)
    metrics.caption_words.labels(mode=mode).observe(len(caption.split()))
    return caption


def caption_image(image: Image.Image, name: str, processor, model, mode: str, client: dict):
    """Caption one image and record the outcome. Returns None if the model fails."""
    try:
        caption = generate_caption(image, processor, model, mode)
    except Exception as e:
        log.error(f"captioning failed for {name}: {e}")
        metrics.errors.labels(mode=mode, error_type="inference_error").inc()
        metrics.images_processed.labels(mode=mode, status="failed", **client).inc()
        return None
    metrics.images_processed.labels(mode=mode, status="success", **client).inc()
    return caption


def open_image(source) -> Image.Image:
    image = Image.open(source)
    image.load()  # force decode now so corrupt files fail here.
    return image


def skip(skipped: list, filename: str, reason: str):
    skipped.append({"file": filename, "reason": SKIP_REASONS[reason]})
    metrics.zip_entries_skipped.labels(reason=reason).inc()


def list_zip_images(archive: zipfile.ZipFile):
    """Split ZIP entries into images to caption and skipped entries with a reason."""
    images, skipped = [], []
    for info in archive.infolist():
        path = PurePosixPath(info.filename)
        if info.is_dir() or path.parts[0] == "__MACOSX" or path.name.startswith("."):
            continue  # folders and OS metadata, not user files.
        if path.suffix.lower().lstrip(".") not in IMAGE_TYPES:
            skip(skipped, info.filename, "not_image")
        elif info.file_size > MAX_IMAGE_BYTES:
            skip(skipped, info.filename, "too_large")
        elif len(images) >= MAX_BULK_IMAGES:
            skip(skipped, info.filename, "over_limit")
        else:
            images.append(info)
    return images, skipped


def caption_zip_entry(archive: zipfile.ZipFile, info: zipfile.ZipInfo, skipped: list, processor, model, client):
    """Caption one ZIP entry. A bad entry is skipped, it must not stop the rest of the batch."""
    try:
        with archive.open(info) as f:
            image = open_image(io.BytesIO(f.read()))
    except Exception as e:
        log.warning(f"could not read {info.filename}: {e}")
        metrics.errors.labels(mode="bulk", error_type="invalid_image").inc()
        skip(skipped, info.filename, "corrupt")
        return None

    start = time.time()
    caption = caption_image(image, info.filename, processor, model, "bulk", client)
    if caption is None:
        skip(skipped, info.filename, "caption_failed")
        return None
    return {"file": info.filename, "caption": caption, "seconds": round(time.time() - start, 2)}


def caption_zip(uploaded, processor, model, client: dict):
    """Caption every image in the ZIP, one at a time. Returns (results, skipped) or None for a bad ZIP."""
    try:
        archive = zipfile.ZipFile(uploaded)
    except zipfile.BadZipFile:
        log.warning(f"{uploaded.name} is not a valid zip file")
        metrics.errors.labels(mode="bulk", error_type="bad_zip").inc()
        st.error(f"`{uploaded.name}` is not a valid ZIP file.")
        return None

    results = []
    with archive:
        images, skipped = list_zip_images(archive)
        metrics.bulk_batch_images.observe(len(images))
        metrics.bulk_images_pending.inc(len(images))
        remaining = len(images)
        progress = st.progress(0.0)
        try:
            for i, info in enumerate(images):
                progress.progress(i / len(images), text=f"Captioning {info.filename} ({i + 1}/{len(images)})")
                result = caption_zip_entry(archive, info, skipped, processor, model, client)
                if result is not None:
                    results.append(result)
                metrics.bulk_images_pending.dec()
                remaining -= 1
        finally:
            # streamlit can interrupt the script mid-batch (e.g. a rerun), release what is left.
            metrics.bulk_images_pending.dec(remaining)
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
        # count each bad upload once, not on every rerun.
        if st.session_state.get("single_bad_upload") != uploaded.file_id:
            st.session_state.single_bad_upload = uploaded.file_id
            metrics.errors.labels(mode="single", error_type="invalid_image").inc()
        return

    st.image(image, caption=uploaded.name, width="stretch")

    if st.button("Generate caption", type="primary", key="single_button"):
        log.info(f"captioning {uploaded.name} ({uploaded.size} bytes)")
        client = client_labels()
        record_request("single", uploaded, client)
        with metrics.active_requests.labels(mode="single").track_inprogress(), \
                metrics.request_seconds.labels(mode="single").time():
            with st.spinner("Generating caption..."):
                caption = caption_image(image, uploaded.name, processor, model, "single", client)
        if caption is None:
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
        client = client_labels()
        record_request("bulk", uploaded, client)
        start = time.time()
        with metrics.active_requests.labels(mode="bulk").track_inprogress(), \
                metrics.request_seconds.labels(mode="bulk").time():
            outcome = caption_zip(uploaded, processor, model, client)
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

start_metrics_server()
init_session_metrics()
processor, model = load_model()

single_tab, bulk_tab = st.tabs(["Single image", "Bulk (ZIP)"])
with single_tab:
    single_mode(processor, model)
with bulk_tab:
    bulk_mode(processor, model)
