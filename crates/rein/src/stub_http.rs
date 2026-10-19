//! Minimal blocking HTTP/1.1 stub for unit tests. Records every request
//! (method, path, headers, body) and answers from a queue of canned
//! `(status, body)` responses, one per connection, then exits. Used to
//! assert the exact auth header / request body our clients send without
//! touching the network.

use std::io::{Read, Write};
use std::net::{TcpListener, TcpStream};
use std::sync::{Arc, Mutex};
use std::time::Duration;

#[derive(Debug, Clone)]
pub(crate) struct StubRequest {
    pub method: String,
    pub path: String,
    pub headers: Vec<(String, String)>,
    pub body: String,
}

impl StubRequest {
    /// Case-insensitive header lookup.
    pub fn header(&self, name: &str) -> Option<&str> {
        self.headers
            .iter()
            .find(|(k, _)| k.eq_ignore_ascii_case(name))
            .map(|(_, v)| v.as_str())
    }

    pub fn json(&self) -> serde_json::Value {
        serde_json::from_str(&self.body).unwrap_or(serde_json::Value::Null)
    }
}

pub(crate) struct StubServer {
    pub url: String,
    requests: Arc<Mutex<Vec<StubRequest>>>,
}

impl StubServer {
    /// Bind 127.0.0.1:0 and serve exactly `responses.len()` connections.
    pub fn start(responses: Vec<(u16, String)>) -> Self {
        let listener = TcpListener::bind("127.0.0.1:0").expect("bind stub");
        let addr = listener.local_addr().expect("stub addr");
        let requests = Arc::new(Mutex::new(Vec::new()));
        let recorded = Arc::clone(&requests);
        std::thread::spawn(move || {
            for (status, body) in responses {
                let Ok((mut stream, _)) = listener.accept() else {
                    break;
                };
                let _ = stream.set_read_timeout(Some(Duration::from_secs(5)));
                let req = read_request(&mut stream);
                recorded.lock().unwrap().push(req);
                let resp = format!(
                    "HTTP/1.1 {status} STUB\r\nContent-Type: application/json\r\n\
                     Content-Length: {}\r\nConnection: close\r\n\r\n{body}",
                    body.len()
                );
                let _ = stream.write_all(resp.as_bytes());
                let _ = stream.flush();
            }
        });
        Self {
            url: format!("http://{addr}"),
            requests,
        }
    }

    pub fn requests(&self) -> Vec<StubRequest> {
        self.requests.lock().unwrap().clone()
    }
}

fn read_request(stream: &mut TcpStream) -> StubRequest {
    let mut buf = Vec::new();
    let mut chunk = [0u8; 4096];
    let head_end = loop {
        let n = match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break None,
            Ok(n) => n,
        };
        buf.extend_from_slice(&chunk[..n]);
        if let Some(pos) = buf.windows(4).position(|w| w == b"\r\n\r\n") {
            break Some(pos + 4);
        }
    };
    let head_end = head_end.unwrap_or(buf.len());
    let head = String::from_utf8_lossy(&buf[..head_end]).into_owned();
    let mut lines = head.split("\r\n");
    let request_line = lines.next().unwrap_or("");
    let mut parts = request_line.split_whitespace();
    let method = parts.next().unwrap_or("").to_string();
    let path = parts.next().unwrap_or("").to_string();
    let mut headers = Vec::new();
    let mut content_length = 0usize;
    for line in lines {
        if let Some((k, v)) = line.split_once(':') {
            let (k, v) = (k.trim().to_string(), v.trim().to_string());
            if k.eq_ignore_ascii_case("content-length") {
                content_length = v.parse().unwrap_or(0);
            }
            headers.push((k, v));
        }
    }
    let mut body = buf[head_end..].to_vec();
    while body.len() < content_length {
        let n = match stream.read(&mut chunk) {
            Ok(0) | Err(_) => break,
            Ok(n) => n,
        };
        body.extend_from_slice(&chunk[..n]);
    }
    StubRequest {
        method,
        path,
        headers,
        body: String::from_utf8_lossy(&body).into_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn records_headers_and_body_and_answers_in_order() {
        let stub = StubServer::start(vec![
            (200, r#"{"ok":true}"#.to_string()),
            (401, r#"{"error":"nope"}"#.to_string()),
        ]);
        let client = reqwest::Client::new();
        let r1 = client
            .post(format!("{}/v1/x", stub.url))
            .header("authorization", "Bearer k1")
            .json(&serde_json::json!({"a": 1}))
            .send()
            .await
            .unwrap();
        assert_eq!(r1.status().as_u16(), 200);
        let r2 = client
            .get(format!("{}/models", stub.url))
            .send()
            .await
            .unwrap();
        assert_eq!(r2.status().as_u16(), 401);
        let reqs = stub.requests();
        assert_eq!(reqs.len(), 2);
        assert_eq!(reqs[0].method, "POST");
        assert_eq!(reqs[0].path, "/v1/x");
        assert_eq!(reqs[0].header("Authorization"), Some("Bearer k1"));
        assert_eq!(reqs[0].json()["a"], 1);
        assert_eq!(reqs[1].method, "GET");
    }
}
