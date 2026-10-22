# Prometheus metrics for the captioning app.
#
# These live in their own module because streamlit reruns app.py on every interaction,
# while a module is imported only once per process, so each metric is registered once.
# NOTE: streamlit re-imports this module if it is edited while the app runs, restart the app after editing.

from prometheus_client import Counter, Gauge, Histogram, Info, Summary
from prometheus_client import disable_created_metrics
from streamlit.runtime import Runtime

# disable **_created metric.
disable_created_metrics()

# ---------- counters ----------
# one per button click, tagged with client metadata.
requests = Counter('caption_requests', 'caption requests (one per button click)',
                   ['mode', 'client_ip', 'session_id', 'browser'])
# images that reached the model, per client and outcome.
images_processed = Counter('caption_images_processed', 'images passed to the captioning model',
                           ['mode', 'client_ip', 'session_id', 'status'])
errors = Counter('caption_errors', 'errors by type (invalid_image, bad_zip, inference_error)',
                 ['mode', 'error_type'])
zip_entries_skipped = Counter('caption_zip_entries_skipped', 'ZIP entries that were not captioned', ['reason'])

# ---------- gauges ----------
active_requests = Gauge('caption_active_requests', 'requests currently being processed', ['mode'])
bulk_images_pending = Gauge('caption_bulk_images_pending', 'images waiting in running bulk jobs')
last_inference = Gauge('caption_last_inference_seconds', 'inference time of the most recent image', ['mode'])
model_loaded = Gauge('caption_model_loaded', '1 once the model is loaded and ready', ['model'])
model_load_seconds = Gauge('caption_model_load_seconds', 'time taken to load the model')
model_size_bytes = Gauge('caption_model_size_bytes', 'memory used by the model parameters (bytes)')
active_sessions = Gauge('caption_active_sessions', 'browser sessions connected to the app')


def _count_active_sessions():
    # streamlit has no public API for this, so read it from the runtime and fall back to 0.
    try:
        return Runtime.instance()._session_mgr.num_active_sessions()
    except Exception:
        return 0


# evaluated on every scrape.
active_sessions.set_function(_count_active_sessions)

# ---------- histograms ----------
# buckets sized for CPU inference, ~1-2s per image is typical.
inference_seconds = Histogram('caption_inference_seconds', 'model.generate time per image (seconds)', ['mode'],
                              buckets=(0.25, 0.5, 0.75, 1, 1.5, 2, 3, 5, 8, 13))
image_pixels = Histogram('caption_image_pixels', 'input image size in pixels (width x height)', ['mode'],
                         buckets=(0.1e6, 0.3e6, 1e6, 2e6, 4e6, 8e6, 16e6, 32e6))
bulk_batch_images = Histogram('caption_bulk_batch_images', 'images to caption per uploaded ZIP',
                              buckets=(1, 5, 10, 20, 30, 40, 50))

# ---------- summaries ----------
# _sum tracks the total, _count tracks the number of observations.
upload_bytes = Summary('caption_upload_bytes', 'size of uploaded files (bytes)', ['mode'])
request_seconds = Summary('caption_request_seconds', 'end-to-end time per request (seconds)', ['mode'])
preprocess_seconds = Summary('caption_preprocess_seconds', 'image preprocessing time, resize + normalise (seconds)',
                             ['mode'])
caption_words = Summary('caption_length_words', 'number of words in the generated caption', ['mode'])

# ---------- info ----------
build = Info('caption_build', 'Prometheus instrumented image captioning app')
