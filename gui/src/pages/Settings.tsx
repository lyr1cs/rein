import { useState, useEffect } from 'react';
import { apiGet } from '../api/client';
import { useDoctor } from '../hooks/useApi';
import type { DoctorCheck, DoctorReport } from '../api/types';

const CATEGORY_LABELS: Record<DoctorCheck['category'], string> = {
  configuration: 'Configuration',
  runtime: 'Runtime',
  storage: 'Storage',
  index: 'Index',
  queue: 'Queue',
  network: 'Network',
};

const SEVERITY_STYLES: Record<DoctorCheck['severity'], string> = {
  info: 'text-[var(--success)] bg-[var(--success)]/10 border-[var(--success)]/20',
  warning: 'text-[var(--warm)] bg-[var(--warm)]/10 border-[var(--warm)]/20',
  error: 'text-[var(--hot)] bg-[var(--hot)]/10 border-[var(--hot)]/20',
};

const STATUS_STYLES: Record<'healthy' | 'degraded' | 'unhealthy', string> = {
  healthy: 'text-[var(--success)]',
  degraded: 'text-[var(--warm)]',
  unhealthy: 'text-[var(--hot)]',
};

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
  const [fixRunning, setFixRunning] = useState(false);
  const [fixError, setFixError] = useState<string | null>(null);
  const { data: doctor, isLoading: doctorLoading, error: doctorError, refetch } = useDoctor();

  // Persist polling interval on change
  useEffect(() => {
    localStorage.setItem('rein_polling_interval', String(pollingInterval));
  }, [pollingInterval]);

  // Persist token on change
  useEffect(() => {
    localStorage.setItem('rein_token', token);
  }, [token]);

  async function runFix() {
    setFixRunning(true);
    setFixError(null);
    try {
      await apiGet<DoctorReport>('/api/doctor?fix=true');
      await refetch();
    } catch (error) {
      setFixError(error instanceof Error ? error.message : 'Failed to run doctor fix');
    } finally {
      setFixRunning(false);
    }
  }

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

        {/* Diagnostics */}
        <div className="bg-[var(--bg-secondary)] border border-[var(--border)] rounded-xl p-5 space-y-4">
          <div className="flex items-start justify-between gap-3">
            <div>
              <div className="text-sm font-medium text-[var(--text-primary)]">Diagnostics</div>
              <div className="text-xs text-[var(--text-muted)] mt-0.5">
                Live output from <span className="font-mono">rein doctor</span>
              </div>
            </div>
            <button
              type="button"
              onClick={() => refetch()}
              className="rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs text-[var(--text-secondary)] hover:text-[var(--text-primary)] hover:bg-[var(--bg-primary)] transition-colors"
            >
              Refresh
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              type="button"
              onClick={runFix}
              disabled={fixRunning}
              className="rounded-lg border border-[var(--accent)]/30 bg-[var(--accent)]/10 px-3 py-1.5 text-xs text-[var(--accent)] transition-colors hover:bg-[var(--accent)]/15 disabled:cursor-not-allowed disabled:opacity-50"
            >
              {fixRunning ? 'Running Fix…' : 'Run Fix'}
            </button>
            <div className="text-[10px] text-[var(--text-muted)]">
              Applies safe local repairs only.
            </div>
          </div>

          {doctorLoading && (
            <div className="text-sm text-[var(--text-muted)]">Loading diagnostics...</div>
          )}

          {doctorError && (
            <div className="rounded-lg border border-[var(--hot)]/20 bg-[var(--hot)]/10 px-3 py-2 text-xs text-[var(--hot)]">
              Failed to load diagnostics: {doctorError instanceof Error ? doctorError.message : 'Unknown error'}
            </div>
          )}

          {fixError && (
            <div className="rounded-lg border border-[var(--hot)]/20 bg-[var(--hot)]/10 px-3 py-2 text-xs text-[var(--hot)]">
              Fix failed: {fixError}
            </div>
          )}

          {doctor && (
            <>
              <div className="flex items-center justify-between rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3">
                <div>
                  <div className="text-xs uppercase tracking-wider text-[var(--text-muted)]">Overall</div>
                  <div className={`text-sm font-medium mt-1 ${STATUS_STYLES[doctor.status]}`}>
                    {doctor.status}
                  </div>
                </div>
                <div className="text-xs text-[var(--text-muted)]">
                  {doctor.checks.length} checks
                </div>
              </div>

              {doctor.fixes_applied && doctor.fixes_applied.length > 0 && (
                <div className="rounded-lg border border-[var(--accent)]/20 bg-[var(--accent)]/8 px-4 py-3 space-y-2">
                  <div className="text-xs uppercase tracking-wider text-[var(--text-muted)]">Fixes Applied</div>
                  <div className="space-y-1">
                    {doctor.fixes_applied.map((fix, index) => (
                      <div key={`${fix}-${index}`} className="text-xs text-[var(--text-secondary)]">
                        {fix}
                      </div>
                    ))}
                  </div>
                </div>
              )}

              <div className="space-y-2">
                {doctor.checks.map((check) => (
                  <div
                    key={check.name}
                    className="rounded-lg border border-[var(--border)] bg-[var(--bg-primary)] px-4 py-3"
                  >
                    <div className="flex items-center gap-2 flex-wrap">
                      <div className="text-sm font-medium text-[var(--text-primary)]">
                        {check.name}
                      </div>
                      <span className="rounded-full border border-[var(--border)] px-2 py-0.5 text-[10px] uppercase tracking-wider text-[var(--text-muted)]">
                        {CATEGORY_LABELS[check.category]}
                      </span>
                      <span className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${SEVERITY_STYLES[check.severity]}`}>
                        {check.severity}
                      </span>
                      {check.fixable && (
                        <span className="rounded-full border border-[var(--accent)]/20 bg-[var(--accent)]/10 px-2 py-0.5 text-[10px] uppercase tracking-wider text-[var(--accent)]">
                          fixable
                        </span>
                      )}
                    </div>
                    <div className="mt-2 text-xs text-[var(--text-secondary)]">
                      {check.message}
                    </div>
                    {check.repair_hint && (
                      <div className="mt-2 rounded-md border border-[var(--border)] bg-[var(--bg-secondary)] px-3 py-2 text-xs text-[var(--text-muted)]">
                        Repair: <span className="text-[var(--text-secondary)]">{check.repair_hint}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </>
          )}
        </div>
      </div>
    </div>
  );
}
