//! Service management: PID files, start/stop, dashboard.

use std::io::Read;
use std::net::{TcpStream, ToSocketAddrs};
use std::path::PathBuf;
use std::time::Duration;

/// Wait for Ctrl-C (all platforms) or SIGTERM (Unix). Completes when either
/// fires. Accept loops use this to stop serving new connections without
/// tearing down spawned per-connection tasks mid-flight.
pub async fn shutdown_signal() {
    #[cfg(unix)]
    {
        use tokio::signal::unix::{signal, SignalKind};
        let mut sigterm = match signal(SignalKind::terminate()) {
            Ok(s) => s,
            Err(e) => {
                tracing::warn!("failed to install SIGTERM handler: {e}");
                let _ = tokio::signal::ctrl_c().await;
                return;
            }
        };
        tokio::select! {
            _ = tokio::signal::ctrl_c() => {}
            _ = sigterm.recv() => {}
        }
    }
    #[cfg(not(unix))]
    {
        let _ = tokio::signal::ctrl_c().await;
    }
}

/// Directory for PID files and runtime state.
fn pid_dir() -> PathBuf {
    let home = std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .unwrap_or_else(|_| "/tmp".to_string());
    PathBuf::from(home).join(".rein")
}

fn pid_path(name: &str) -> PathBuf {
    pid_dir().join(format!("{name}.pid"))
}

fn effective_pid_name(name: &str) -> &str {
    name
}

/// Write PID + exe path to `~/.rein/{name}.pid` for identity verification.
pub fn write_pid(name: &str) -> anyhow::Result<()> {
    write_pid_of(name, std::process::id())
}

/// Write an arbitrary PID + current exe path to `~/.rein/{name}.pid`.
pub fn write_pid_of(name: &str, pid: u32) -> anyhow::Result<()> {
    let dir = pid_dir();
    std::fs::create_dir_all(&dir)?;
    let exe = std::env::current_exe().unwrap_or_default();
    let content = format!("{}\n{}", pid, exe.display());
    std::fs::write(pid_path(effective_pid_name(name)), content)?;
    Ok(())
}

/// Remove PID file on shutdown.
pub fn remove_pid(name: &str) {
    let _ = std::fs::remove_file(pid_path(effective_pid_name(name)));
}

fn matches_recorded_executable(running: &std::path::Path, saved_exe: &str) -> bool {
    let saved_path = std::path::Path::new(saved_exe);
    if saved_exe.trim().is_empty() {
        return running.file_name().is_some_and(|name| name == "rein");
    }
    running == saved_path || running.file_name() == saved_path.file_name()
}

/// Check whether the process at `pid` is actually the recorded rein binary.
/// Uses OS-specific introspection to guard against PID recycling.
fn is_process_rein(pid: u32, saved_exe: &str) -> bool {
    #[cfg(target_os = "linux")]
    {
        if let Ok(exe) = std::fs::read_link(format!("/proc/{pid}/exe")) {
            return matches_recorded_executable(&exe, saved_exe);
        }
    }
    #[cfg(target_os = "macos")]
    {
        if let Ok(output) = std::process::Command::new("ps")
            .args(["-p", &pid.to_string(), "-o", "comm="])
            .output()
        {
            if output.status.success() {
                let command = String::from_utf8_lossy(&output.stdout);
                return matches_recorded_executable(
                    std::path::Path::new(command.trim()),
                    saved_exe,
                );
            }
        }
    }
    // If we can't determine, assume it's still rein (conservative)
    true
}

/// Read PID from file, verify it's still a rein process, and return PID if alive.
pub fn is_running(name: &str) -> Option<u32> {
    let content = std::fs::read_to_string(pid_path(effective_pid_name(name))).ok()?;
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
        // Verify the running process is actually rein (guards against PID recycling)
        if !is_process_rein(pid, saved_exe) {
            remove_pid(name);
            return None;
        }
        Some(pid)
    }
    #[cfg(not(unix))]
    {
        let _ = (pid, saved_exe, name);
        remove_pid(name);
        None
    }
}

/// Check if a port is responding.
fn probe_host_for_bind(bind: &str) -> &str {
    match bind {
        "0.0.0.0" | "::" | "localhost" => "127.0.0.1",
        "::1" => "[::1]",
        other => other,
    }
}

fn display_host_for_bind(bind: &str) -> &str {
    match bind {
        "0.0.0.0" | "::" => "127.0.0.1",
        other => other,
    }
}

fn port_is_open(bind: &str, port: u16) -> bool {
    let addr = format!("{}:{port}", probe_host_for_bind(bind));
    addr.to_socket_addrs()
        .ok()
        .and_then(|mut addrs| addrs.next())
        .is_some_and(|addr| TcpStream::connect_timeout(&addr, Duration::from_millis(200)).is_ok())
}

/// Stop a service by sending SIGTERM.
pub fn stop_service(name: &str) -> anyhow::Result<()> {
    match is_running(name) {
        Some(pid) => {
            #[cfg(not(unix))]
            {
                anyhow::bail!("service stop is only supported on Unix platforms");
            }
            #[cfg(unix)]
            {
                unsafe {
                    libc::kill(pid as i32, libc::SIGTERM);
                }
                // Wait briefly for process to exit
                for _ in 0..20 {
                    std::thread::sleep(Duration::from_millis(100));
                    if is_running(name).is_none() {
                        remove_pid(name);
                        println!("Stopped {name} (PID {pid})");
                        return Ok(());
                    }
                }
                anyhow::bail!("{name} did not stop within timeout (PID {pid})");
            }
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
    // Write PID file immediately to prevent race with concurrent start attempts
    let _ = write_pid_of(name, pid);
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
    // Prefer REIN_PROXY_TOKEN (matches proxy auth) then REIN_HTTP_TOKEN as fallback
    let token = std::env::var("REIN_PROXY_TOKEN")
        .ok()
        .filter(|token| !token.trim().is_empty())
        .or_else(|| {
            std::env::var("REIN_HTTP_TOKEN")
                .ok()
                .filter(|token| !token.trim().is_empty())
        });
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
        port_open: port_is_open(&config.server.sse_bind, gui_port),
    };
    let proxy_status = ServiceStatus {
        name: "Proxy",
        pid: is_running("proxy"),
        port: proxy_port,
        port_open: port_is_open(&config.proxy.bind, proxy_port),
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
        let bind = display_host_for_bind(&config.server.sse_bind);
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

    // Queue status — scan the DB-scoped queue subdir for memory_queue*.jsonl files
    let db_tag = {
        use std::hash::{Hash, Hasher};
        let mut h = std::collections::hash_map::DefaultHasher::new();
        config.resolve_db_path().hash(&mut h);
        format!("{:016x}", h.finish())
    };
    let buffer_dir = crate::extract::hooks::buffer::resolve_buffer_dir(config)
        .join("queue")
        .join(&db_tag);
    let pending = std::fs::read_dir(&buffer_dir)
        .map(|entries| {
            entries
                .filter_map(|e| e.ok())
                .filter(|e| {
                    let name = e.file_name();
                    let name = name.to_string_lossy();
                    name.starts_with("memory_queue") && name.ends_with(".jsonl")
                })
                .filter(|e| e.metadata().map(|m| m.len() > 0).unwrap_or(false))
                .count()
        })
        .unwrap_or(0);
    println!("\nQueue: {pending} pending");
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn explicit_pid_names_are_not_rewritten() {
        assert_eq!(effective_pid_name("gui"), "gui");
        assert_eq!(effective_pid_name("http"), "http");
        assert_eq!(effective_pid_name("proxy"), "proxy");
    }
}
