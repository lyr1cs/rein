//! Service management: PID files, start/stop, dashboard.

use std::io::Read;
use std::net::TcpStream;
use std::path::PathBuf;
use std::time::Duration;

/// Directory for PID files and runtime state.
fn pid_dir() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_default();
    PathBuf::from(home).join(".rein")
}

fn pid_path(name: &str) -> PathBuf {
    pid_dir().join(format!("{name}.pid"))
}

/// Write PID + exe path to `~/.rein/{name}.pid` for identity verification.
pub fn write_pid(name: &str) -> anyhow::Result<()> {
    let dir = pid_dir();
    std::fs::create_dir_all(&dir)?;
    let exe = std::env::current_exe().unwrap_or_default();
    let content = format!("{}\n{}", std::process::id(), exe.display());
    std::fs::write(pid_path(name), content)?;
    Ok(())
}

/// Remove PID file on shutdown.
pub fn remove_pid(name: &str) {
    let _ = std::fs::remove_file(pid_path(name));
}

/// Read PID from file, verify it's still a rein process, and return PID if alive.
pub fn is_running(name: &str) -> Option<u32> {
    let content = std::fs::read_to_string(pid_path(name)).ok()?;
    let mut lines = content.lines();
    let pid: u32 = lines.next()?.trim().parse().ok()?;
    let saved_exe = lines.next().unwrap_or("");

    #[cfg(unix)]
    {
        // Check if process is alive
        if unsafe { libc::kill(pid as i32, 0) } != 0 {
            // Process dead — stale PID file
            remove_pid(name);
            return None;
        }
        // Verify it's actually a rein process by checking /proc or lsof
        // On macOS, check exe path via sysctl; simplest portable check: compare saved exe
        if !saved_exe.is_empty() {
            if let Ok(current_exe) = std::env::current_exe() {
                if !saved_exe.contains("rein") && current_exe.to_string_lossy().contains("rein") {
                    // Saved exe doesn't look like rein — PID was recycled
                    remove_pid(name);
                    return None;
                }
            }
        }
        Some(pid)
    }
    #[cfg(not(unix))]
    {
        let _ = saved_exe;
        Some(pid)
    }
}

/// Check if a port is responding.
fn port_is_open(port: u16) -> bool {
    TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(200),
    )
    .is_ok()
}

/// Stop a service by sending SIGTERM.
pub fn stop_service(name: &str) -> anyhow::Result<()> {
    match is_running(name) {
        Some(pid) => {
            #[cfg(unix)]
            unsafe {
                libc::kill(pid as i32, libc::SIGTERM);
            }
            // Wait briefly for process to exit
            for _ in 0..20 {
                std::thread::sleep(Duration::from_millis(100));
                if is_running(name).is_none() {
                    break;
                }
            }
            remove_pid(name);
            println!("Stopped {name} (PID {pid})");
            Ok(())
        }
        None => {
            // Maybe stale PID file
            remove_pid(name);
            println!("{name} is not running");
            Ok(())
        }
    }
}

/// Start a service by spawning the current binary in background.
pub fn start_service(name: &str, serve_args: &[&str]) -> anyhow::Result<()> {
    if let Some(pid) = is_running(name) {
        println!("{name} is already running (PID {pid})");
        return Ok(());
    }

    let exe = std::env::current_exe()?;
    let child = std::process::Command::new(exe)
        .args(serve_args)
        .stdin(std::process::Stdio::null())
        .stdout(std::process::Stdio::null())
        .stderr(std::process::Stdio::null())
        .spawn()?;

    let pid = child.id();
    println!("Started {name} (PID {pid})");

    // Wait a moment and check if the process is still alive (catches immediate failures).
    std::thread::sleep(Duration::from_millis(500));

    #[cfg(unix)]
    {
        let ret = unsafe { libc::kill(pid as i32, 0) };
        if ret != 0 {
            anyhow::bail!("{name} failed to start — check logs");
        }
    }

    Ok(())
}

// ---------------------------------------------------------------------------
// Dashboard
// ---------------------------------------------------------------------------

struct ServiceStatus {
    name: &'static str,
    pid: Option<u32>,
    port: u16,
    port_open: bool,
}

impl ServiceStatus {
    fn status_indicator(&self) -> &'static str {
        match (self.pid, self.port_open) {
            (Some(_), true) => "\x1b[32m●\x1b[0m", // green
            (None, true) => "\x1b[33m●\x1b[0m",    // yellow (port open, no PID file)
            _ => "\x1b[31m●\x1b[0m",               // red
        }
    }

    fn status_text(&self) -> String {
        match (self.pid, self.port_open) {
            (Some(pid), true) => format!("running  :{:<5} PID {pid}", self.port),
            (None, true) => format!("running  :{:<5} (external)", self.port),
            (Some(pid), false) => format!("stale    :{:<5} PID {pid} (port closed)", self.port),
            (None, false) => format!("stopped  :{}", self.port),
        }
    }
}

/// Fetch proxy metrics from the metrics endpoint.
fn fetch_proxy_metrics(port: u16) -> Option<(u64, u64, u64)> {
    let mut stream = TcpStream::connect_timeout(
        &std::net::SocketAddr::from(([127, 0, 0, 1], port)),
        Duration::from_millis(300),
    )
    .ok()?;
    // Include auth token if configured (proxy requires x-rein-token when token is set)
    let token = std::env::var("REIN_PROXY_TOKEN")
        .ok()
        .or_else(|| std::env::var("REIN_HTTP_TOKEN").ok());
    let auth_header = token
        .map(|t| format!("x-rein-token: {t}\r\n"))
        .unwrap_or_default();
    let request = format!(
        "GET /rein/metrics HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{auth_header}Connection: close\r\n\r\n"
    );
    std::io::Write::write_all(&mut stream, request.as_bytes()).ok()?;
    let mut response = String::new();
    stream.read_to_string(&mut response).ok()?;
    // Parse JSON from response body (after headers)
    let body = response.split("\r\n\r\n").nth(1)?;
    let json: serde_json::Value = serde_json::from_str(body).ok()?;
    let requests = json.get("request_count")?.as_u64()?;
    let errors = json.get("error_count")?.as_u64()?;
    let extractions = json.get("extraction_count")?.as_u64()?;
    Some((requests, errors, extractions))
}

/// Print the dashboard.
pub fn print_dashboard(config: &crate::config::ReinConfig) {
    use crate::types::traits::MemoryStore;
    let version = env!("CARGO_PKG_VERSION");
    let gui_port = config.server.sse_port;
    let proxy_port = config.proxy.port;

    let gui_status = ServiceStatus {
        name: "GUI",
        pid: is_running("gui"),
        port: gui_port,
        port_open: port_is_open(gui_port),
    };
    let proxy_status = ServiceStatus {
        name: "Proxy",
        pid: is_running("proxy"),
        port: proxy_port,
        port_open: port_is_open(proxy_port),
    };

    println!("rein v{version} dashboard\n");

    // Services
    println!("Services");
    println!(
        "  {:<12} {} {}",
        gui_status.name,
        gui_status.status_indicator(),
        gui_status.status_text()
    );
    if gui_status.port_open {
        let bind = &config.server.sse_bind;
        println!("  {:<12}   http://{bind}:{gui_port}/", "");
    }
    println!(
        "  {:<12} {} {}",
        proxy_status.name,
        proxy_status.status_indicator(),
        proxy_status.status_text()
    );

    // Proxy metrics
    if proxy_status.port_open {
        if let Some((requests, errors, extractions)) = fetch_proxy_metrics(proxy_port) {
            println!("\nProxy");
            println!(
                "  Requests: {}  Errors: {}  Extractions: {}",
                requests, errors, extractions
            );
        }
    }

    // Memory stats
    if let Ok(store) = config.open_store() {
        if let Ok(stats) = store.stats() {
            println!("\nMemory");
            println!(
                "  Total: {}  LTM: {}  STM: {}",
                stats.total_memories, stats.ltm_count, stats.stm_count
            );
            println!(
                "  Concepts: {}  Topics: {}  Memoirs: {}",
                stats.concept_count, stats.topic_count, stats.memoir_count
            );
        }
    }

    // Queue status — scan the buffer dir for memory_queue_*.jsonl files
    let buffer_dir = crate::extract::hooks::buffer::resolve_buffer_dir(config);
    let pending = std::fs::read_dir(&buffer_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.starts_with("memory_queue_") && name.ends_with(".jsonl")
                })
                .filter(|e| e.metadata().map(|m| m.len() > 0).unwrap_or(false))
                .count()
        })
        .unwrap_or(0);
    println!("\nQueue: {pending} pending");
}
