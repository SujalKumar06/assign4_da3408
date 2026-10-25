A coding agent (Claude Code) was used in the development of this assignment.

- It was used to generate the python files, `app.py` and `metrics.py`
- The design of the grafana dashboard was designed using it - `caption_app_dashboard.json`
- The PromQL in `rules.yml` (recording rules and alerts) was generated using it.
  - Prompt (condensed from the conversation): "Write a `rules.yml` with recording rules for CPU utilisation, app throughput and p95 inference latency, and alerts for the app being down, high inference latency, a high error rate, high CPU and high memory, each with a `severity` label. Route the alerts through Alertmanager to a webhook." The agent also added `NodeExporterDown` and `RootDiskAlmostFull`.

| Rule | PromQL | What it computes |
|---|---|---|
| `instance:node_cpu_busy:percent` | `100 * (1 - avg by (instance) (rate(node_cpu_seconds_total{mode="idle"}[30s])))` | CPU busy % across all cores (100 minus the idle share) |
| `instance:node_memory_used:percent` | `100 * (1 - node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes)` | RAM in use % |
| `instance:node_root_disk_used:percent` | `100 * (1 - node_filesystem_avail_bytes{mountpoint="/"} / node_filesystem_size_bytes{mountpoint="/"})` | `/` filesystem used % |
| `mode:caption_images:rate1m` | `sum by (mode) (rate(caption_images_processed_total[1m]))` | images captioned per second per mode |
| `mode:caption_inference_seconds:p95_1m` | `histogram_quantile(0.95, sum by (le, mode) (rate(caption_inference_seconds_bucket[1m])))` | p95 inference time per mode from the histogram buckets |
| `mode:caption_errors:rate1m` | `sum by (mode) (rate(caption_errors_total[1m]))` | errors per second per mode |
| `HighErrorRate` | `sum(rate(caption_errors_total[2m])) / (sum(rate(caption_images_processed_total[2m])) + sum(rate(caption_errors_total[2m]))) > 0.2` | more than 20% of caption attempts failing over 2 minutes (no traffic gives NaN, which never fires) |
