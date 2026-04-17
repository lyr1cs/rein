//! WebSocket frame mirror state machine.
//!
//! Extracted from `proxy/mod.rs` in v0.19.0 to isolate the ~270-line raw
//! RFC 6455 frame decoder. The mirror runs alongside the transparent WS
//! relay and records extracted text (redacted, length-bounded) into a
//! per-session artifact. It is strictly record-only — nothing here mutates
//! what rein forwards to the upstream or client.
//!
//! ## Defensive contract
//!
//! * **Per-message inflate cap** ([`WebSocketMirrorState::MAX_INFLATED_BYTES`],
//!   1 MiB): deflate bomb protection. Permessage-deflate payloads that would
//!   expand past the cap are dropped and logged, not allocated.
//! * **Per-frame length cap** (`MAX_WS_FRAME_BYTES`, 16 MiB): attacker-
//!   controlled 64-bit extended length fields are rejected before any slice
//!   is built. On rejection, `pending` is drained and fragmentation state
//!   is cleared so the mirror returns to a known-good baseline.
//! * **Checked arithmetic** on `offset + payload_len` prevents pathological
//!   wraparound on 32-bit targets.
//! * **Protocol-violation state reset** on out-of-sequence frames (new text
//!   frame mid-fragmentation, orphan continuation, close) — stale bytes from
//!   a partial fragment never bleed into the next message.

use tokio_tungstenite::tungstenite::Message;

use super::{responses, truncate_utf8_to_byte_limit};

#[derive(Default)]
pub(super) struct WebSocketMirrorState {
    /// Raw bytes awaiting frame-boundary decode. Only consumed by the test-
    /// only [`Self::feed`] path; unused under production [`Self::record_message`].
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) pending: Vec<u8>,
    /// Partial fragment accumulator for the raw-frame decoder (test-only).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fragmented_payload: Option<Vec<u8>>,
    /// `rsv1` bit of the open fragmentation start frame (test-only).
    #[cfg_attr(not(test), allow(dead_code))]
    pub(super) fragmented_compressed: bool,
    pub(super) event_messages: Vec<String>,
    pub(super) assistant_text: String,
    pub(super) request_query: Option<String>,
    pub(super) event_bytes: usize,
    pub(super) truncated: bool,
    pub(super) close_seen: bool,
}

impl WebSocketMirrorState {
    pub(super) fn record_message(&mut self, message: &Message, collect_assistant_text: bool) {
        match message {
            Message::Text(text) => {
                self.record_text_message(text.to_string(), collect_assistant_text, 256_000);
            }
            Message::Close(_) => {
                self.close_seen = true;
                self.push_event_limited("[close]".to_string(), 256_000);
            }
            _ => {}
        }
    }

    /// Bound per-message inflate output so an attacker-controlled upstream
    /// frame cannot expand a tiny compressed payload (deflate bomb) into
    /// unbounded memory. 1 MiB is generous for legitimate `/responses` deltas
    /// — real messages are typically <50 KB.
    ///
    /// NOTE (v0.19.0 finding, corrected by v0.19.1 Codex review):
    /// this cap is only reached through the raw-frame entry point
    /// [`Self::feed`], which is currently used only by unit tests. Production
    /// WS relay ([`relay_websocket_with_mirror`] in `mod.rs`) calls
    /// [`Self::record_message`] with a pre-decoded `tungstenite::Message`.
    ///
    /// The actual production defence against permessage-deflate bombs is
    /// NOT done by tungstenite (tokio-tungstenite 0.28 does not support
    /// permessage-deflate and rejects RSV1 frames as `NonZeroReservedBits`).
    /// It comes from rein's handshake rewrite — we strip the
    /// `sec-websocket-extensions` header in [`should_strip_ws_handshake_header`]
    /// so compression is never negotiated end-to-end. If a future change
    /// restores extension passthrough (or wires up a compressed transport),
    /// the bomb cap MUST be reconnected into the production path and a
    /// real compressed-frame E2E test added.
    #[cfg(test)]
    pub(super) const MAX_INFLATED_BYTES: u64 = 1024 * 1024;

    #[cfg(test)]
    pub(super) fn decode_text_payload(payload: &[u8], compressed: bool) -> Option<String> {
        if compressed {
            let mut data = payload.to_vec();
            data.extend_from_slice(&[0x00, 0x00, 0xff, 0xff]);
            let decoder = flate2::read::DeflateDecoder::new(&data[..]);
            let mut limited = std::io::Read::take(decoder, Self::MAX_INFLATED_BYTES + 1);
            let mut output = Vec::new();
            use std::io::Read as _;
            limited.read_to_end(&mut output).ok()?;
            if output.len() as u64 > Self::MAX_INFLATED_BYTES {
                tracing::warn!(
                    limit = Self::MAX_INFLATED_BYTES,
                    got = output.len(),
                    "websocket mirror: permessage-deflate payload exceeded inflate cap, dropping"
                );
                return None;
            }
            String::from_utf8(output).ok()
        } else {
            Some(String::from_utf8_lossy(payload).into_owned())
        }
    }

    /// Parse raw WebSocket bytes into frames and record text payloads.
    ///
    /// Currently used only from unit tests that synthesize RFC 6455 frames
    /// directly (see `websocket_mirror_*` tests in `mod.rs`). Production
    /// relay consumes `tungstenite::Message`s via [`Self::record_message`].
    #[cfg(test)]
    pub(super) fn feed(&mut self, chunk: &[u8], collect_assistant_text: bool) {
        const MAX_EVENT_BYTES: usize = 256_000;
        self.pending.extend_from_slice(chunk);
        loop {
            if self.pending.len() < 2 {
                return;
            }
            let b0 = self.pending[0];
            let b1 = self.pending[1];
            let fin = (b0 & 0x80) != 0;
            let rsv1 = (b0 & 0x40) != 0;
            let opcode = b0 & 0x0f;
            let masked = (b1 & 0x80) != 0;
            let mut offset = 2usize;
            // Hard cap on a single frame. RFC 6455 allows up to 2^63-1 bytes
            // via the 64-bit extended length, but in practice no sane client
            // or server sends multi-GB frames and a malicious upstream could
            // construct huge values to crash or stall the mirror. We cap at
            // 16 MiB per frame — 16× the per-message inflate cap — which
            // still comfortably covers real Codex `/responses` deltas and
            // any reasonable batched update.
            const MAX_WS_FRAME_BYTES: u64 = 16 * 1024 * 1024;
            let payload_len: usize = match b1 & 0x7f {
                len @ 0..=125 => len as usize,
                126 => {
                    if self.pending.len() < offset + 2 {
                        return;
                    }
                    let len = u16::from_be_bytes([self.pending[offset], self.pending[offset + 1]])
                        as usize;
                    offset += 2;
                    len
                }
                127 => {
                    if self.pending.len() < offset + 8 {
                        return;
                    }
                    let len64 = u64::from_be_bytes([
                        self.pending[offset],
                        self.pending[offset + 1],
                        self.pending[offset + 2],
                        self.pending[offset + 3],
                        self.pending[offset + 4],
                        self.pending[offset + 5],
                        self.pending[offset + 6],
                        self.pending[offset + 7],
                    ]);
                    offset += 8;
                    // Reject obviously-malicious frame lengths: anything above
                    // MAX_WS_FRAME_BYTES, or anything that can't fit in usize
                    // on a 32-bit platform. We drain `pending` and treat it as
                    // an oversize frame; this bounds per-frame work to O(cap).
                    if len64 > MAX_WS_FRAME_BYTES {
                        tracing::warn!(
                            len64,
                            cap = MAX_WS_FRAME_BYTES,
                            "websocket mirror: extended frame length exceeds cap; discarding buffer"
                        );
                        self.pending.clear();
                        self.fragmented_payload = None;
                        self.fragmented_compressed = false;
                        return;
                    }
                    match usize::try_from(len64) {
                        Ok(len) => len,
                        Err(_) => {
                            tracing::warn!(
                                len64,
                                "websocket mirror: extended length exceeds usize on this target"
                            );
                            self.pending.clear();
                            self.fragmented_payload = None;
                            self.fragmented_compressed = false;
                            return;
                        }
                    }
                }
                _ => unreachable!(),
            };
            let mask = if masked {
                if self.pending.len() < offset + 4 {
                    return;
                }
                let mask = [
                    self.pending[offset],
                    self.pending[offset + 1],
                    self.pending[offset + 2],
                    self.pending[offset + 3],
                ];
                offset += 4;
                Some(mask)
            } else {
                None
            };
            // Guard against offset + payload_len overflow (pathological inputs).
            let total = match offset.checked_add(payload_len) {
                Some(total) => total,
                None => {
                    tracing::warn!(
                        offset,
                        payload_len,
                        "websocket mirror: offset + payload_len overflowed; discarding buffer"
                    );
                    self.pending.clear();
                    self.fragmented_payload = None;
                    self.fragmented_compressed = false;
                    return;
                }
            };
            if self.pending.len() < total {
                return;
            }
            let mut payload = self.pending[offset..total].to_vec();
            self.pending.drain(..total);
            if let Some(mask) = mask {
                for (index, byte) in payload.iter_mut().enumerate() {
                    *byte ^= mask[index % 4];
                }
            }

            match opcode {
                0x1 => {
                    // Protocol violation: a new text frame while fragmentation
                    // is in progress. Clear any partial fragment so the next
                    // continuation doesn't see stale bytes.
                    if self.fragmented_payload.is_some() {
                        tracing::warn!(
                            "websocket mirror: new text frame arrived mid-fragmentation; \
                            discarding partial payload"
                        );
                        self.fragmented_payload = None;
                        self.fragmented_compressed = false;
                    }
                    if fin {
                        if let Some(text) = Self::decode_text_payload(&payload, rsv1) {
                            self.record_text_message(text, collect_assistant_text, MAX_EVENT_BYTES);
                        } else {
                            self.push_event_limited(
                                "[compressed websocket frame decode failed]".to_string(),
                                MAX_EVENT_BYTES,
                            );
                        }
                    } else {
                        self.fragmented_payload = Some(payload);
                        self.fragmented_compressed = rsv1;
                    }
                }
                0x0 => {
                    if let Some(existing) = self.fragmented_payload.as_mut() {
                        existing.extend_from_slice(&payload);
                        if fin {
                            let payload = self.fragmented_payload.take().unwrap_or_default();
                            let compressed = self.fragmented_compressed;
                            self.fragmented_compressed = false;
                            if let Some(text) = Self::decode_text_payload(&payload, compressed) {
                                self.record_text_message(
                                    text,
                                    collect_assistant_text,
                                    MAX_EVENT_BYTES,
                                );
                            } else {
                                self.push_event_limited(
                                    "[compressed websocket frame decode failed]".to_string(),
                                    MAX_EVENT_BYTES,
                                );
                            }
                        }
                    } else {
                        // Orphan continuation with no prior start frame:
                        // discard and log.
                        tracing::debug!(
                            "websocket mirror: orphan continuation frame with no active fragment"
                        );
                    }
                }
                0x8 => {
                    self.close_seen = true;
                    // Clear any in-progress fragment on close to avoid leaking
                    // state across potential reconnects of the same mirror.
                    self.fragmented_payload = None;
                    self.fragmented_compressed = false;
                }
                _ => {}
            }
        }
    }

    fn record_text_message(
        &mut self,
        text: String,
        collect_assistant_text: bool,
        max_event_bytes: usize,
    ) {
        let text = crate::extract::hooks::parsing::redact_secrets(&text);
        if !collect_assistant_text && self.request_query.is_none() {
            self.request_query = responses::extract_query_ws_message(&text)
                .map(|query| query.trim().to_string())
                .filter(|query| !query.is_empty());
        }
        self.push_event_limited(text.clone(), max_event_bytes);
        if collect_assistant_text {
            if let Some(delta) = responses::extract_assistant_text_ws_message(&text) {
                // Cap assistant_text the same way stream_response caps MAX_EXTRACT_BUF
                // in mod.rs — without this, a long-lived /responses WS session with a
                // misbehaving upstream can grow the buffer without bound and OOM the
                // proxy. 200 KB comfortably covers a single assistant turn and mirrors
                // the SSE path's policy.
                const MAX_ASSISTANT_TEXT_BYTES: usize = 200_000;
                if self.assistant_text.len() >= MAX_ASSISTANT_TEXT_BYTES {
                    self.truncated = true;
                } else {
                    let remaining = MAX_ASSISTANT_TEXT_BYTES - self.assistant_text.len();
                    if delta.len() > remaining {
                        let preview = truncate_utf8_to_byte_limit(&delta, remaining);
                        self.assistant_text.push_str(&preview);
                        self.truncated = true;
                    } else {
                        self.assistant_text.push_str(&delta);
                    }
                }
            }
        }
    }

    fn push_event(&mut self, text: String) {
        self.event_bytes += text.len();
        self.event_messages.push(text);
    }

    pub(super) fn push_event_limited(&mut self, text: String, max_event_bytes: usize) {
        if self.event_bytes >= max_event_bytes {
            self.truncated = true;
            return;
        }
        let remaining = max_event_bytes - self.event_bytes;
        if text.len() > remaining {
            let preview = truncate_utf8_to_byte_limit(&text, remaining);
            self.push_event(preview);
            self.truncated = true;
            return;
        }
        self.push_event(text);
    }
}
