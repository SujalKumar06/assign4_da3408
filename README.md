# AI Operations (AIOps): Assignment 4

Structure of the repository:

```
app.py                              Streamlit app: single image and bulk (ZIP) captioning, /metrics on :18000
metrics.py                          every Prometheus metric the app exposes
requirements.txt                    Python dependencies (CPU-only torch)
prometheus.yml                      scrape config (app, node_exporter, Prometheus) and alerting
rules.yml                           recording rules and alert rules
alertmanager.yml                    Alertmanager route and webhook receiver
grafana/caption_app_dashboard.json  Grafana dashboard export
.streamlit/config.toml              Streamlit settings (file watcher off)
```

| Service | Port |
|---|---|
| Streamlit app | 8501 |
| App metrics (`/metrics`) | 18000 |
| Prometheus | 9090 |
| Alertmanager | 9093 |
| node_exporter | 9100 |
| Grafana | 3000 |

## Environment

```bash
conda create -n aiops python=3.13 -y
conda activate aiops
pip install -r requirements.txt
```

Also needed: [Prometheus](https://prometheus.io/download/) (3.15.0),
[node_exporter](https://prometheus.io/download/#node_exporter) (1.12.1),
[Alertmanager](https://prometheus.io/download/#alertmanager) (0.34.1) and Docker
for Grafana. Run every command from the repository root.

I extracted Prometheus, node_exporter and Alertmanager into my `~/Downloads`
folder, which is why the commands below start with `~/Downloads/...`. If yours
are somewhere else, replace that part with your own path.

## Running

Steps 1 to 4 each run in their own terminal.

1. node_exporter

   ```bash
   ~/Downloads/node_exporter/node_exporter
   ```

2. Alertmanager (set your own webhook URL in `alertmanager.yml` first)

   ```bash
   ~/Downloads/alertmanager/alertmanager --config.file=alertmanager.yml --storage.path=alertmanager-data
   ```

3. Prometheus

   ```bash
   ~/Downloads/prometheus/prometheus --config.file=prometheus.yml
   ```

4. The app

   ```bash
   streamlit run app.py
   ```

   Open <http://localhost:8501>. The metrics server starts on the first page
   load, so the app shows as DOWN in Prometheus until then.

5. Grafana

   ```bash
   docker run -d --name grafana -p 3000:3000 grafana/grafana-enterprise
   ```

   At <http://localhost:3000>, add a Prometheus data source with URL
   `http://172.17.0.1:9090` (`http://host.docker.internal:9090` on Docker
   Desktop), then Dashboards → New → Import `grafana/caption_app_dashboard.json`.

Check: <http://localhost:9090/targets> shows `caption_app`, `node` and
`prometheus` as UP.

## Metrics

| Type | Metric | What it tracks |
|---|---|---|
| Counter | `caption_requests_total` | requests per mode, client IP, session and browser |
| Counter | `caption_images_processed_total` | images through the model per mode, client and status (success / failed) |
| Counter | `caption_errors_total` | errors per mode and type (`invalid_image`, `bad_zip`, `inference_error`) |
| Counter | `caption_zip_entries_skipped_total` | ZIP entries not captioned, per reason |
| Gauge | `caption_active_requests` | requests in flight per mode |
| Gauge | `caption_bulk_images_pending` | images waiting in running bulk jobs |
| Gauge | `caption_last_inference_seconds` | inference time of the latest image per mode |
| Gauge | `caption_model_loaded` | 1 once the model is ready |
| Gauge | `caption_model_load_seconds` | model load time |
| Gauge | `caption_model_size_bytes` | memory of the model weights |
| Gauge | `caption_active_sessions` | connected browser sessions |
| Histogram | `caption_inference_seconds` | per-image model time per mode |
| Histogram | `caption_image_pixels` | input resolution per mode |
| Histogram | `caption_bulk_batch_images` | images per ZIP |
| Summary | `caption_upload_bytes` | uploaded bytes per mode |
| Summary | `caption_request_seconds` | end-to-end request time per mode |
| Summary | `caption_preprocess_seconds` | resize + normalise time per image |
| Summary | `caption_length_words` | words per generated caption |
| Info | `caption_build` | version, model, device, torch version |

## Alerts

| Alert | Severity | Fires when |
|---|---|---|
| `CaptionAppDown` | critical | the app's `/metrics` is unreachable for 10 s |
| `NodeExporterDown` | critical | node_exporter is unreachable for 10 s |
| `HighInferenceLatency` | warning | p95 inference time above 3 s for 30 s |
| `HighErrorRate` | warning | more than 20 % of caption attempts fail (2 min window) |
| `HighCPU` | warning | machine CPU busy above 80 % for 15 s |
| `HighMemory` | warning | machine RAM used above 90 % for 30 s |
| `RootDiskAlmostFull` | warning | `/` above 95 % full for 1 min |
