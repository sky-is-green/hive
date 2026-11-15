//! Native daemon metrics and streaming observations.
//!
//! Prometheus is the operational contract: the daemon renders its own HTTP
//! counters/latencies plus per-stream time-to-first-token, and forwards the
//! healthy llama-server samples with `hive_` labels. The older
//! `/v1/server/metrics` JSON route remains for the console.

use std::collections::BTreeMap;
use std::pin::Pin;
use std::sync::Mutex;
use std::sync::atomic::{AtomicU64, Ordering};
use std::task::{Context, Poll};
use std::time::{Duration, Instant};

use bytes::Bytes;
use futures_util::Stream;
use serde_json::Value;

const REQUEST_BUCKETS: &[f64] = &[
    0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 300.0,
];
const TTFT_BUCKETS: &[f64] = &[0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0];

struct Histogram {
    bounds: &'static [f64],
    buckets: Vec<u64>,
    sum: f64,
    count: u64,
}

impl Histogram {
    fn new(bounds: &'static [f64]) -> Self {
        Self {
            bounds,
            buckets: vec![0; bounds.len() + 1],
            sum: 0.0,
            count: 0,
        }
    }

    fn observe(&mut self, value: f64) {
        let index = self
            .bounds
            .iter()
            .position(|bound| value <= *bound)
            .unwrap_or(self.buckets.len() - 1);
        self.buckets[index] += 1;
        self.sum += value;
        self.count += 1;
    }

    fn render(&self, out: &mut String, name: &str) {
        let mut cumulative = 0;
        for (index, bound) in self.bounds.iter().enumerate() {
            cumulative += self.buckets[index];
            out.push_str(&format!("{name}_bucket{{le=\"{bound}\"}} {cumulative}\n"));
        }
        cumulative += self.buckets[self.bounds.len()];
        out.push_str(&format!("{name}_bucket{{le=\"+Inf\"}} {cumulative}\n"));
        out.push_str(&format!("{name}_sum {:.6}\n", self.sum));
        out.push_str(&format!("{name}_count {}\n", self.count));
    }
}

pub struct Metrics {
    started: Instant,
    requests: Mutex<BTreeMap<(String, String, u16), u64>>,
    request_duration: Mutex<Histogram>,
    stream_ttft: Mutex<Histogram>,
    stream_deltas: AtomicU64,
    streams: AtomicU64,
}

impl Default for Metrics {
    fn default() -> Self {
        Self::new()
    }
}

impl Metrics {
    pub fn new() -> Self {
        Self {
            started: Instant::now(),
            requests: Mutex::new(BTreeMap::new()),
            request_duration: Mutex::new(Histogram::new(REQUEST_BUCKETS)),
            stream_ttft: Mutex::new(Histogram::new(TTFT_BUCKETS)),
            stream_deltas: AtomicU64::new(0),
            streams: AtomicU64::new(0),
        }
    }

    pub fn observe_request(&self, method: &str, route: &str, status: u16, elapsed: Duration) {
        self.requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .entry((method.to_owned(), route.to_owned(), status))
            .and_modify(|count| *count += 1)
            .or_insert(1);
        self.request_duration
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .observe(elapsed.as_secs_f64());
    }

    pub fn observe_stream(&self) {
        self.streams.fetch_add(1, Ordering::Relaxed);
    }

    pub fn observe_ttft(&self, elapsed: Duration) {
        self.stream_ttft
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .observe(elapsed.as_secs_f64());
    }

    pub fn add_stream_deltas(&self, count: u64) {
        self.stream_deltas.fetch_add(count, Ordering::Relaxed);
    }

    pub fn render(&self, engine_samples: &[String], instances: &[Value]) -> String {
        let mut out = String::with_capacity(4096);
        out.push_str("# HELP hive_up 1 when the daemon process is serving\n");
        out.push_str("# TYPE hive_up gauge\nhive_up 1\n");
        out.push_str("# HELP hive_uptime_seconds Daemon process uptime\n");
        out.push_str("# TYPE hive_uptime_seconds gauge\n");
        out.push_str(&format!(
            "hive_uptime_seconds {:.3}\n",
            self.started.elapsed().as_secs_f64()
        ));
        if let Some((rss_bytes, threads)) = process_memory() {
            out.push_str("# HELP hive_process_resident_memory_bytes Resident memory\n");
            out.push_str("# TYPE hive_process_resident_memory_bytes gauge\n");
            out.push_str(&format!("hive_process_resident_memory_bytes {rss_bytes}\n"));
            out.push_str("# HELP hive_process_threads OS threads\n");
            out.push_str("# TYPE hive_process_threads gauge\n");
            out.push_str(&format!("hive_process_threads {threads}\n"));
        }

        out.push_str("# HELP hive_http_requests_total HTTP requests by route and status\n");
        out.push_str("# TYPE hive_http_requests_total counter\n");
        for ((method, route, status), count) in self
            .requests
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .iter()
        {
            out.push_str(&format!(
                "hive_http_requests_total{{method=\"{}\",route=\"{}\",status=\"{}\"}} {}\n",
                escape_label(method),
                escape_label(route),
                status,
                count
            ));
        }

        out.push_str("# HELP hive_http_request_duration_seconds HTTP request latency\n");
        out.push_str("# TYPE hive_http_request_duration_seconds histogram\n");
        self.request_duration
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .render(&mut out, "hive_http_request_duration_seconds");

        out.push_str("# HELP hive_stream_ttft_seconds Time to first proxied stream chunk\n");
        out.push_str("# TYPE hive_stream_ttft_seconds histogram\n");
        self.stream_ttft
            .lock()
            .unwrap_or_else(|error| error.into_inner())
            .render(&mut out, "hive_stream_ttft_seconds");
        out.push_str("# HELP hive_stream_delta_events_total SSE delta events proxied\n");
        out.push_str("# TYPE hive_stream_delta_events_total counter\n");
        out.push_str(&format!(
            "hive_stream_delta_events_total {}\n",
            self.stream_deltas.load(Ordering::Relaxed)
        ));
        out.push_str("# HELP hive_streams_total Streaming proxy connections\n");
        out.push_str("# TYPE hive_streams_total counter\n");
        out.push_str(&format!(
            "hive_streams_total {}\n",
            self.streams.load(Ordering::Relaxed)
        ));

        out.push_str("# HELP hive_engine_up Whether a managed engine is reachable\n");
        out.push_str("# TYPE hive_engine_up gauge\n");
        for instance in instances {
            let tier = instance
                .get("key")
                .and_then(Value::as_str)
                .filter(|value| !value.is_empty())
                .or_else(|| instance.get("role").and_then(Value::as_str))
                .unwrap_or("");
            let model = instance.get("model").and_then(Value::as_str).unwrap_or("");
            let port = instance.get("port").and_then(Value::as_u64).unwrap_or(0);
            let healthy = instance
                .get("healthy")
                .and_then(Value::as_bool)
                .unwrap_or(false);
            out.push_str(&format!(
                "hive_engine_up{{tier=\"{}\",model=\"{}\",port=\"{}\"}} {}\n",
                escape_label(tier),
                escape_label(model),
                port,
                u8::from(healthy)
            ));
            if let Some(pid) = instance.get("pid").and_then(Value::as_u64) {
                out.push_str(&format!(
                    "hive_engine_pid{{tier=\"{}\",model=\"{}\",port=\"{}\"}} {}\n",
                    escape_label(tier),
                    escape_label(model),
                    port,
                    pid
                ));
            }
        }
        for sample in engine_samples {
            out.push_str(sample);
            if !sample.ends_with('\n') {
                out.push('\n');
            }
        }
        out
    }
}

/// Prefix a llama-server sample with `hive_` and append routing labels.
pub fn prefix_engine_sample(line: &str, tier: &str, model: &str, port: u16) -> Option<String> {
    let line = line.trim();
    if line.is_empty() || line.starts_with('#') {
        return None;
    }
    let (sample, value) = line.rsplit_once(' ')?;
    let value = value.trim().parse::<f64>().ok()?;
    let (name, labels) = match sample.split_once('{') {
        Some((name, rest)) => {
            let labels = rest.strip_suffix('}').unwrap_or(rest);
            (name, labels)
        }
        None => (sample, ""),
    };
    if name.is_empty() {
        return None;
    }
    let extra = format!(
        "tier=\"{}\",model=\"{}\",port=\"{}\"",
        escape_label(tier),
        escape_label(model),
        port
    );
    let labels = if labels.is_empty() {
        extra
    } else {
        format!("{labels},{extra}")
    };
    Some(format!("hive_{name}{{{labels}}} {value}\n"))
}

/// Map a request path to a bounded-cardinality route label.
pub fn route_label(path: &str) -> String {
    if let Some(rest) = path.strip_prefix("/v1/memory/") {
        let mut parts = rest.splitn(2, '/');
        let _conversation = parts.next().unwrap_or("");
        return match parts.next() {
            Some(action) if !action.is_empty() => {
                format!("/v1/memory/:conversation/{action}")
            }
            _ => "/v1/memory/:conversation".to_owned(),
        };
    }
    if let Some(rest) = path.strip_prefix("/v1/stacks/") {
        let mut parts = rest.splitn(2, '/');
        let _name = parts.next().unwrap_or("");
        return match parts.next() {
            Some(action) if !action.is_empty() => format!("/v1/stacks/:name/{action}"),
            _ => "/v1/stacks/:name".to_owned(),
        };
    }
    if path.starts_with("/v1/models/hub/files/") {
        return "/v1/models/hub/files/:repo".to_owned();
    }
    if path.starts_with("/v1/report/") {
        return "/v1/report/:run".to_owned();
    }
    path.to_owned()
}

fn escape_label(value: &str) -> String {
    value
        .replace('\\', "\\\\")
        .replace('"', "\\\"")
        .replace('\n', "\\n")
}

fn process_memory() -> Option<(u64, u64)> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let mut rss_kib = 0u64;
    for line in status.lines() {
        if let Some(value) = line.strip_prefix("VmRSS:") {
            rss_kib = value.split_whitespace().next()?.parse().ok()?;
        }
    }
    let threads = std::fs::read_dir("/proc/self/task")
        .map(|entries| entries.count() as u64)
        .unwrap_or(0);
    Some((rss_kib * 1024, threads))
}

/// A proxied byte stream that records time-to-first-byte and SSE delta events.
pub struct ObservedStream<S> {
    inner: S,
    metrics: std::sync::Arc<Metrics>,
    started: Instant,
    first: bool,
    tail: Vec<u8>,
    scanned: usize,
}

impl<S> ObservedStream<S> {
    pub fn new(inner: S, metrics: std::sync::Arc<Metrics>) -> Self {
        metrics.observe_stream();
        Self {
            inner,
            metrics,
            started: Instant::now(),
            first: false,
            tail: Vec::new(),
            scanned: 0,
        }
    }

    fn count_deltas(&mut self, chunk: &[u8]) {
        const NEEDLES: [&[u8]; 2] = [b"\"type\": \"delta\"", b"\"type\":\"delta\""];
        self.tail.extend_from_slice(chunk);
        let mut count = 0u64;
        for needle in NEEDLES {
            let mut start = self.scanned;
            while start + needle.len() <= self.tail.len() {
                if &self.tail[start..start + needle.len()] == needle {
                    count += 1;
                }
                start += 1;
            }
        }
        if count > 0 {
            self.metrics.add_stream_deltas(count);
        }
        let max_len = NEEDLES.iter().map(|needle| needle.len()).max().unwrap_or(1);
        self.scanned = self.tail.len().saturating_sub(max_len - 1);
        // Retain enough overlap for a needle split across network chunks.
        if self.tail.len() > 4096 {
            let keep = self.tail.len() - 32;
            self.tail.drain(..keep);
            self.scanned = 0;
        }
    }
}

impl<S, E> Stream for ObservedStream<S>
where
    S: Stream<Item = Result<Bytes, E>> + Unpin,
{
    type Item = Result<Bytes, E>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        let this = &mut *self;
        match Pin::new(&mut this.inner).poll_next(cx) {
            Poll::Ready(Some(Ok(bytes))) => {
                if !this.first {
                    this.first = true;
                    this.metrics.observe_ttft(this.started.elapsed());
                }
                this.count_deltas(&bytes);
                Poll::Ready(Some(Ok(bytes)))
            }
            Poll::Ready(Some(Err(error))) => Poll::Ready(Some(Err(error))),
            Poll::Ready(None) => Poll::Ready(None),
            Poll::Pending => Poll::Pending,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn route_labels_stay_bounded() {
        assert_eq!(
            route_label("/v1/memory/conv-123/assemble"),
            "/v1/memory/:conversation/assemble"
        );
        assert_eq!(
            route_label("/v1/stacks/live/apply"),
            "/v1/stacks/:name/apply"
        );
        assert_eq!(
            route_label("/v1/models/hub/files/org/repo"),
            "/v1/models/hub/files/:repo"
        );
        assert_eq!(route_label("/health"), "/health");
    }

    #[test]
    fn prometheus_render_includes_requests_and_engines() {
        let metrics = Metrics::new();
        metrics.observe_request("GET", "/health", 200, Duration::from_millis(2));
        metrics.observe_ttft(Duration::from_millis(80));
        metrics.add_stream_deltas(3);
        let text = metrics.render(
            &[
                prefix_engine_sample("llama_tokens_predicted_total 42", "face", "smol", 1234)
                    .unwrap(),
            ],
            &[serde_json::json!({
                "key": "face",
                "model": "smol",
                "port": 1234,
                "pid": 99,
                "healthy": true
            })],
        );
        assert!(text.contains("hive_up 1"));
        assert!(text.contains(
            "hive_http_requests_total{method=\"GET\",route=\"/health\",status=\"200\"} 1"
        ));
        assert!(text.contains("hive_stream_ttft_seconds_count 1"));
        assert!(text.contains("hive_stream_delta_events_total 3"));
        assert!(text.contains("hive_engine_up{tier=\"face\",model=\"smol\",port=\"1234\"} 1"));
        assert!(text.contains(
            "hive_llama_tokens_predicted_total{tier=\"face\",model=\"smol\",port=\"1234\"} 42"
        ));
    }
}
