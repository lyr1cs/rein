import { useState, useEffect } from 'react';

/* ── Settings page ──────────────────────────────────────────────── */

export default function Settings() {
  const [pollingInterval, setPollingInterval] = useState(() => {
    const saved = localStorage.getItem('rein_polling_interval');
    return saved ? parseInt(saved, 10) : 5;
  });

  const [token, setToken] = useState(() => {
    return localStorage.getItem('rein_token') || '';
  });

  const [showToken, setShowToken] = useState(false);

  // Persist polling interval on change
  useEffect(() => {
    localStorage.setItem('rein_polling_interval', String(pollingInterval));
  }, [pollingInterval]);

  // Persist token on change
  useEffect(() => {
    localStorage.setItem('rein_token', token);
  }, [token]);

  return (
    <div className="flex flex-col h-full overflow-y-auto">
      <div className="max-w-2xl mx-auto w-full px-6 py-8 space-y-8">
        {/* Page header */}
        <div>
          <h2 className="text-lg font-medium text-[var(--text-primary)]">Settings</h2>
          <p className="text-xs text-[var(--text-muted)] mt-1">
            Changes are saved automatically to localStorage.
          </p>
        </div>

        {/* Polling Interval */}
        <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <div>
              <div className="text-sm font-medium text-[var(--text-primary)]">Polling Interval</div>
              <div className="text-xs text-[var(--text-muted)] mt-0.5">
                How often to refresh data from the server
              </div>
            </div>
            <div className="text-sm font-mono text-[var(--accent)]">
              {pollingInterval}s
            </div>
          </div>
          <input
            type="range"
            min={1}
            max={60}
            value={pollingInterval}
            onChange={(e) => setPollingInterval(parseInt(e.target.value, 10))}
            className="w-full h-1.5 rounded-full appearance-none cursor-pointer"
            style={{
              background: `linear-gradient(to right, var(--accent) 0%, var(--accent) ${((pollingInterval - 1) / 59) * 100}%, var(--border) ${((pollingInterval - 1) / 59) * 100}%, var(--border) 100%)`,
            }}
          />
          <div className="flex justify-between text-xs text-[var(--text-muted)]">
            <span>1s</span>
            <span>30s</span>
            <span>60s</span>
          </div>
        </div>

        {/* Auth Token */}
        <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-5 space-y-3">
          <div>
            <div className="text-sm font-medium text-[var(--text-primary)]">Auth Token</div>
            <div className="text-xs text-[var(--text-muted)] mt-0.5">
              Bearer token for authenticating with the rein HTTP API
            </div>
          </div>
          <div className="relative">
            <input
              type={showToken ? 'text' : 'password'}
              value={token}
              onChange={(e) => setToken(e.target.value)}
              placeholder="Enter your REIN_HTTP_TOKEN..."
              className="w-full bg-[var(--bg-primary)] border border-[var(--border)] rounded-lg pl-4 pr-12 py-2 text-sm text-[var(--text-primary)] placeholder-[var(--text-muted)] focus:outline-none focus:border-[var(--accent)] transition-colors font-mono"
            />
            <button
              onClick={() => setShowToken(!showToken)}
              className="absolute right-2 top-1/2 -translate-y-1/2 px-2 py-1 text-xs text-[var(--text-muted)] hover:text-[var(--text-secondary)] transition-colors rounded"
              type="button"
            >
              {showToken ? 'Hide' : 'Show'}
            </button>
          </div>
        </div>

        {/* About */}
        <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-5 space-y-3">
          <div className="text-sm font-medium text-[var(--text-primary)]">About</div>
          <div className="space-y-2">
            <div className="text-sm text-[var(--text-secondary)]">
              rein Neural Wiki v0.10.3
            </div>
            <div className="text-xs text-[var(--text-muted)]">
              Multi-source cross-validated memory MCP server with adaptive engine,
              temporal knowledge graph, and autonomous retrieval routing.
            </div>
            <a
              href="https://github.com/lyr1cs/rein"
              target="_blank"
              rel="noopener noreferrer"
              className="inline-flex items-center gap-1.5 text-xs text-[var(--accent)] hover:underline mt-1"
            >
              GitHub Repository
              <span className="text-[10px]">{'\u2197'}</span>
            </a>
          </div>
        </div>
      </div>
    </div>
  );
}
