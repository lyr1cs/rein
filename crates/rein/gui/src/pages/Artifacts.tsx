import { useState, useCallback, useEffect, useRef } from 'react';
import { useArtifacts } from '../hooks/useApi';
import { apiGet } from '../api/client';
import type { Artifact, ArtifactDetail, Turn } from '../api/types';

/* ── helpers ─────────────────────────────────────────────────────── */

function formatDate(iso: string): string {
  return new Date(iso).toLocaleDateString('en-US', {
    month: 'short',
    day: 'numeric',
    year: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  });
}

function parseTranscript(text: string): Turn[] {
  const turns: Turn[] = [];
  const lines = text.split('\n');
  let currentRole = '';
  let currentText = '';

  for (const line of lines) {
    const match = line.match(/^(User|Assistant|user|assistant|Human|human):\s*(.*)/i);
    if (match) {
      if (currentRole) {
        turns.push({ role: currentRole, text: currentText.trim() });
      }
      currentRole = match[1].toLowerCase() === 'human' ? 'user' : match[1].toLowerCase();
      currentText = match[2];
    } else {
      currentText += '\n' + line;
    }
  }
  if (currentRole) {
    turns.push({ role: currentRole, text: currentText.trim() });
  }

  // If no roles parsed, return the whole thing as a single block
  if (turns.length === 0 && text.trim()) {
    turns.push({ role: 'transcript', text: text.trim() });
  }

  return turns;
}

/* ── Artifacts page ─────────────────────────────────────────────── */

export default function Artifacts() {
  const [limit, setLimit] = useState(20);
  const [expandedId, setExpandedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<ArtifactDetail | null>(null);
  const [detailLoading, setDetailLoading] = useState(false);
  const [detailError, setDetailError] = useState<string | null>(null);
  const detailAbortRef = useRef<AbortController | null>(null);
  const detailRequestRef = useRef(0);

  const { data, isLoading } = useArtifacts(limit);
  const artifacts = data?.artifacts ?? [];

  useEffect(() => () => detailAbortRef.current?.abort(), []);

  const handleRowClick = useCallback(
    async (artifact: Artifact) => {
      if (expandedId === artifact.id) {
        detailAbortRef.current?.abort();
        detailRequestRef.current += 1;
        setExpandedId(null);
        setDetail(null);
        setDetailError(null);
        setDetailLoading(false);
        return;
      }
      detailAbortRef.current?.abort();
      const controller = new AbortController();
      detailAbortRef.current = controller;
      detailRequestRef.current += 1;
      const requestId = detailRequestRef.current;
      setExpandedId(artifact.id);
      setDetail(null);
      setDetailError(null);
      setDetailLoading(true);
      try {
        const d = await apiGet<ArtifactDetail>(
          `/api/artifacts/${artifact.id}?include_transcript=true`,
          { signal: controller.signal },
        );
        if (detailRequestRef.current === requestId && !controller.signal.aborted) {
          setDetail(d);
        }
      } catch (err) {
        if (!(err instanceof DOMException && err.name === 'AbortError')) {
          console.error('Failed to fetch artifact detail:', err);
          if (detailRequestRef.current === requestId) {
            setDetailError(err instanceof Error ? err.message : 'Failed to load transcript');
          }
        }
      } finally {
        if (detailRequestRef.current === requestId) {
          setDetailLoading(false);
        }
      }
    },
    [expandedId],
  );

  const handleLoadMore = useCallback(() => {
    setLimit((prev) => prev + 20);
  }, []);

  return (
    <div className="flex flex-col h-full overflow-hidden">
      {/* Header */}
      <div className="px-6 pt-5 pb-4 flex items-center justify-between">
        <h2 className="text-lg font-medium text-[var(--text-primary)]">Session Artifacts</h2>
        <span className="text-xs text-[var(--text-muted)]">
          {isLoading ? 'Loading...' : `${artifacts.length} artifacts`}
        </span>
      </div>

      {/* Table */}
      <div className="flex-1 overflow-y-auto px-6 pb-6">
        {isLoading && artifacts.length === 0 ? (
          <div className="flex items-center justify-center py-20 text-[var(--text-muted)]">
            Loading...
          </div>
        ) : artifacts.length === 0 ? (
          <div className="flex flex-col items-center justify-center py-20 text-[var(--text-muted)]">
            <div className="text-3xl mb-3">{'\uD83D\uDCC4'}</div>
            <div className="text-sm">No artifacts found</div>
          </div>
        ) : (
          <div className="space-y-0">
            {/* Table header */}
            <div className="grid grid-cols-[1fr_120px_80px_120px_160px] gap-3 px-4 py-2 text-xs text-[var(--text-muted)] uppercase tracking-wider border-b border-[var(--border)]">
              <div>Title</div>
              <div>Agent</div>
              <div>Turns</div>
              <div>Episode</div>
              <div>Created</div>
            </div>

            {/* Rows */}
            {artifacts.map((artifact) => (
              <div key={artifact.id}>
                <button
                  onClick={() => handleRowClick(artifact)}
                  className={`w-full grid grid-cols-[1fr_120px_80px_120px_160px] gap-3 px-4 py-3 text-left transition-colors hover:bg-[var(--bg-secondary)] border-b border-[var(--border)] cursor-pointer ${
                    expandedId === artifact.id ? 'bg-[var(--bg-secondary)]' : ''
                  }`}
                >
                  <div className="text-sm text-[var(--text-primary)] truncate">
                    {artifact.title || 'Untitled'}
                  </div>
                  <div className="text-xs text-[var(--text-secondary)] truncate">
                    {artifact.source_agent || '-'}
                  </div>
                  <div className="text-xs text-[var(--text-secondary)] font-mono">
                    {artifact.turn_count}
                  </div>
                  <div className="text-xs text-[var(--text-muted)] font-mono truncate">
                    {artifact.episode_id ? artifact.episode_id.slice(0, 8) : '-'}
                  </div>
                  <div className="text-xs text-[var(--text-muted)]">
                    {formatDate(artifact.created_at)}
                  </div>
                </button>

                {/* Expanded detail */}
                {expandedId === artifact.id && (
                  <div className="bg-[var(--bg-secondary)] border-b border-[var(--border)] px-6 py-4">
                    {detailLoading ? (
                      <div className="text-xs text-[var(--text-muted)] py-4">
                        Loading transcript...
                      </div>
                    ) : detailError ? (
                      <div className="text-xs text-[var(--hot)] py-2 break-words">
                        Failed to load transcript: {detailError}
                      </div>
                    ) : detail?.transcript_text ? (
                      <div className="space-y-3 max-h-96 overflow-y-auto">
                        {parseTranscript(detail.transcript_text).map((turn, i) => (
                          <div
                            key={i}
                            className={`rounded-lg p-3 ${
                              turn.role === 'user'
                                ? 'bg-[var(--accent)]/10 border border-[var(--accent)]/20'
                                : turn.role === 'assistant'
                                ? 'bg-[var(--bg-primary)] border border-[var(--border)]'
                                : 'bg-[var(--bg-primary)] border border-[var(--border)]'
                            }`}
                          >
                            <div className="text-xs text-[var(--text-muted)] uppercase tracking-wider mb-1.5">
                              {turn.role === 'user' ? 'User' : turn.role === 'assistant' ? 'Assistant' : 'Transcript'}
                            </div>
                            <div className="text-sm text-[var(--text-secondary)] leading-relaxed whitespace-pre-wrap break-words">
                              {turn.text}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="text-xs text-[var(--text-muted)] py-2">
                        No transcript available
                      </div>
                    )}

                    {/* Metadata row */}
                    {detail && (
                      <div className="flex items-center gap-4 mt-3 pt-3 border-t border-[var(--border)]">
                        {detail.artifact_kind && (
                          <span className="text-xs px-2 py-0.5 rounded bg-[var(--accent)]/15 text-[var(--accent)]">
                            {detail.artifact_kind}
                          </span>
                        )}
                        {detail.source_label && (
                          <span className="text-xs text-[var(--text-muted)]">
                            {detail.source_label}
                          </span>
                        )}
                        {detail.summary && (
                          <span className="text-xs text-[var(--text-secondary)] truncate">
                            {detail.summary}
                          </span>
                        )}
                      </div>
                    )}
                  </div>
                )}
              </div>
            ))}

            {/* Load More */}
            <div className="flex justify-center py-4">
              <button
                onClick={handleLoadMore}
                className="px-6 py-2 text-xs rounded-lg bg-[var(--bg-secondary)] border border-[var(--border)] text-[var(--text-secondary)] hover:border-[var(--accent)]/50 hover:text-[var(--accent)] transition-colors"
              >
                Load More
              </button>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
