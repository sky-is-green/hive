//! Minimal HTTP/1.1 GET for loopback engine probes.
//!
//! The supervisor and `hive status` only need JSON from servers they already
//! trust on localhost. Keeping this in core avoids pulling a TLS stack into a
//! binary the gate measures for size. HTTPS URLs report a clear error.

use std::io::{Read, Write};
use std::net::{TcpStream, ToSocketAddrs};
use std::time::Duration;

#[derive(Debug)]
pub struct Response {
    pub status: u16,
    pub reason: String,
    pub body: Vec<u8>,
}

impl Response {
    pub fn json(&self) -> Result<serde_json::Value, String> {
        serde_json::from_slice(&self.body).map_err(|error| error.to_string())
    }
}

pub fn get(url: &str, timeout: Duration) -> Result<Response, String> {
    let endpoint = Endpoint::parse(url)?;
    let addresses: Vec<_> = (endpoint.host.as_str(), endpoint.port)
        .to_socket_addrs()
        .map_err(|error| error.to_string())?
        .collect();
    let mut stream = None;
    let mut last_error = String::from("no address");
    for address in addresses {
        match TcpStream::connect_timeout(&address, timeout) {
            Ok(candidate) => {
                stream = Some(candidate);
                break;
            }
            Err(error) => last_error = error.to_string(),
        }
    }
    let Some(mut stream) = stream else {
        return Err(last_error);
    };
    let _ = stream.set_read_timeout(Some(timeout));
    let _ = stream.set_write_timeout(Some(timeout));
    let host = if endpoint.port == 80 {
        endpoint.host.clone()
    } else {
        format!("{}:{}", endpoint.host, endpoint.port)
    };
    let request = format!(
        "GET {} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\nAccept: application/json\r\n\r\n",
        endpoint.path
    );
    stream
        .write_all(request.as_bytes())
        .map_err(|error| error.to_string())?;
    let mut raw = Vec::new();
    stream
        .take(1 << 20)
        .read_to_end(&mut raw)
        .map_err(|error| error.to_string())?;
    let text = String::from_utf8_lossy(&raw);
    let Some((headers, body)) = text.split_once("\r\n\r\n") else {
        return Err("malformed HTTP response".into());
    };
    let status_line = headers.lines().next().unwrap_or_default();
    let mut parts = status_line.splitn(3, ' ');
    let _version = parts.next();
    let status = parts
        .next()
        .and_then(|code| code.parse::<u16>().ok())
        .ok_or_else(|| "malformed HTTP status".to_owned())?;
    let reason = parts.next().unwrap_or("").trim().to_owned();
    Ok(Response {
        status,
        reason,
        body: body.as_bytes().to_vec(),
    })
}

/// An `http://host:port/path` URL split into its parts.
struct Endpoint {
    host: String,
    port: u16,
    path: String,
}

impl Endpoint {
    fn parse(url: &str) -> Result<Self, String> {
        let lower = url.to_ascii_lowercase();
        let rest = if let Some(rest) = lower.strip_prefix("http://") {
            &url[url.len() - rest.len()..]
        } else if lower.starts_with("https://") {
            return Err("https endpoints are not supported by the local probe".into());
        } else {
            return Err("unsupported endpoint URL".into());
        };
        let (authority, path) = match rest.split_once('/') {
            Some((authority, path)) => (authority, format!("/{path}")),
            None => (rest, "/".to_owned()),
        };
        let (host, port) = if let Some(stripped) = authority.strip_prefix('[') {
            let end = stripped
                .find(']')
                .ok_or_else(|| "malformed IPv6 endpoint".to_owned())?;
            let host = stripped[..end].to_owned();
            let port = stripped[end + 1..]
                .strip_prefix(':')
                .and_then(|port| port.parse::<u16>().ok())
                .unwrap_or(80);
            (host, port)
        } else {
            match authority.rsplit_once(':') {
                Some((host, port)) if port.chars().all(|c| c.is_ascii_digit()) => {
                    (host.to_owned(), port.parse::<u16>().unwrap_or(80))
                }
                _ => (authority.to_owned(), 80),
            }
        };
        if host.is_empty() {
            return Err("endpoint has no host".into());
        }
        Ok(Self { host, port, path })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_loopback_urls() {
        let endpoint = Endpoint::parse("http://127.0.0.1:8080/v1/models").unwrap();
        assert_eq!(endpoint.host, "127.0.0.1");
        assert_eq!(endpoint.port, 8080);
        assert_eq!(endpoint.path, "/v1/models");
    }

    #[test]
    fn rejects_https() {
        assert!(Endpoint::parse("https://example.com/v1/models").is_err());
    }
}
