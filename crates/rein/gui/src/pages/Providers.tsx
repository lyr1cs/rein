import { useCallback, useEffect, useRef, useState } from 'react';
import { useQueryClient } from '@tanstack/react-query';
import { apiPost } from '../api/client';
import type { ProviderStatusReport } from '../api/types';
import { useProviders } from '../hooks/useApi';
import FlowDiagram from './providers/FlowDiagram';
import KeyCard from './providers/KeyCard';
import RoleCard from './providers/RoleCard';
import { ROLE_COPY } from './providers/copy';
import { credentialName, groupByRole, roleStatus, summarize } from './providers/providersModel';

const REVEAL_TTL_MS = 60_000;

function describeError(error: unknown): string {
  const message = error instanceof Error ? error.message : String(error);
  if (message.includes('REIN_HTTP_TOKEN') || message.includes('owner')) {
    return 'Reveal is owner-only. Apply REIN_HTTP_TOKEN in Settings first.';
  }
  if (message.includes('protected-read') || message.includes('401')) {
    return 'This page needs the read token. Apply REIN_HTTP_TOKEN in Settings, or switch [server].auth away from "public".';
  }
  return message;
}

export default function Providers() {
  const [probe, setProbe] = useState(false);
  const { data, isLoading, error, refetch, isFetching } = useProviders(probe);
  const queryClient = useQueryClient();
  const [selectedRole, setSelectedRole] = useState<string | null>(null);
  const [selectedKey, setSelectedKey] = useState<string | null>(null);
  const [revealed, setRevealed] = useState<Record<string, { value: string; until: number }>>({});
  const [revealErrors, setRevealErrors] = useState<Record<string, string>>({});
  const timers = useRef<Record<string, number>>({});
  const [showTables, setShowTables] = useState(false);

  // Plaintext lives only here; cleared on unmount and after the TTL.
  useEffect(() => {
    const t = timers.current;
    return () => {
      Object.values(t).forEach((id) => window.clearTimeout(id));
    };
  }, []);

  const hide = useCallback((id: string) => {
    window.clearTimeout(timers.current[id]);
    delete timers.current[id];
    setRevealed((r) => {
      const next = { ...r };
      delete next[id];
      return next;
    });
  }, []);

  const toggleReveal = useCallback(async (id: string) => {
    if (revealed[id]) {
      hide(id);
      return;
    }
    setRevealErrors((e) => ({ ...e, [id]: '' }));
    try {
      const res = await apiPost<{ id: string; value: string }>('/api/providers/reveal', { credential: id });
      setRevealed((r) => ({ ...r, [id]: { value: res.value, until: Date.now() + REVEAL_TTL_MS } }));
      timers.current[id] = window.setTimeout(() => hide(id), REVEAL_TTL_MS);
    } catch (err) {
      setRevealErrors((e) => ({ ...e, [id]: describeError(err) }));
    }
  }, [revealed, hide]);

  const jumpToRole = (role: string) => {
    setSelectedRole(role);
    document.getElementById(`role-${role}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };
  const jumpToKey = (id: string) => {
    setSelectedKey(id);
    document.getElementById(`key-${credentialName(id)}`)?.scrollIntoView({ behavior: 'smooth', block: 'center' });
  };

  const runProbe = async () => {
    setProbe(true);
    await queryClient.invalidateQueries({ queryKey: ['providers', true] });
  };

  if (isLoading) {
    return <div className="flex h-full items-center justify-center text-[var(--text-muted)]">Loading AI keys...</div>;
  }
  if (error || !data) {
    return (
      <div className="p-6">
        <h1 className="mb-4 text-xl font-semibold text-[var(--text-primary)]">AI Keys</h1>
        <div className="text-sm text-[var(--hot)]">{describeError(error ?? new Error('no data'))}</div>
      </div>
    );
  }
  const report: ProviderStatusReport = data;
  const groups = groupByRole(report);
  const sum = summarize(report);
  const keyCreds = report.credentials.filter((c) => c.kind !== 'endpoint');

  return (
    <div className="mx-auto max-w-6xl space-y-6 p-6">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <h1 className="text-xl font-semibold text-[var(--text-primary)]">AI Keys</h1>
          <div className="mt-1 text-xs text-[var(--text-muted)]">
            {sum.presentKeys}/{sum.keys} keys present · {sum.readyRoles} roles ready · {sum.missingRoles} missing a key · {sum.disabledRoles} off
            {report.probed_at && <span> · probed {new Date(report.probed_at).toLocaleTimeString()}</span>}
          </div>
        </div>
        <div className="flex items-center gap-2">
          <button type="button" onClick={() => refetch()} className="rounded-lg border border-[var(--border)] px-3 py-1.5 text-xs text-[var(--text-secondary)] hover:bg-[var(--bg-primary)]">Refresh</button>
          <button type="button" onClick={runProbe} disabled={isFetching} className="rounded-lg border border-[var(--accent)]/30 bg-[var(--accent)]/10 px-3 py-1.5 text-xs text-[var(--accent)] hover:bg-[var(--accent)]/15 disabled:opacity-50">
            {isFetching && probe ? 'Testing…' : 'Test keys'}
          </button>
        </div>
      </div>

      <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
        <div className="mb-2 text-xs uppercase tracking-wider text-[var(--text-muted)]">How rein uses AI</div>
        <FlowDiagram report={report} onSelectRole={jumpToRole} />
      </div>

      {report.vector_index && (
        <div className={`rounded-lg border px-4 py-3 text-xs ${report.vector_index.matches ? 'border-[var(--border)] text-[var(--text-secondary)]' : 'border-[var(--warm)]/30 bg-[var(--warm)]/10 text-[var(--warm)]'}`}>
          vector index: stamped <span className="font-mono">{report.vector_index.stamped ?? 'none'}</span> · configured <span className="font-mono">{report.vector_index.configured}</span> · {report.vector_index.rows.toLocaleString()} rows ·{' '}
          {report.vector_index.matches ? '✓ matches' : <>✗ does not match — run <code>rein warmup --reembed-all</code></>}
        </div>
      )}

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
        {groups.map((g) => (
          <RoleCard key={g.role} role={g.role} surfaces={g.surfaces} status={roleStatus(g.surfaces)} report={report} highlighted={selectedRole === g.role} onSelectKey={jumpToKey} />
        ))}
      </div>

      <div>
        <h2 className="mb-3 text-sm font-semibold uppercase tracking-wider text-[var(--text-muted)]">Keys</h2>
        {keyCreds.length === 0 ? (
          <div className="rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-5 text-sm text-[var(--text-secondary)]">
            No API keys configured — rein is running keyword-only / local models. Pick a vendor with <code>rein providers vendors</code>.
          </div>
        ) : (
          <div className="grid grid-cols-1 gap-4 md:grid-cols-2">
            {keyCreds.map((c) => (
              <KeyCard key={c.id} credential={c} report={report} revealed={revealed[c.id]?.value ?? null} revealError={revealErrors[c.id] || null} onToggleReveal={() => toggleReveal(c.id)} highlighted={selectedKey === c.id} onSelectRole={jumpToRole} />
            ))}
          </div>
        )}
      </div>

      <div>
        <button type="button" onClick={() => setShowTables(!showTables)} className="text-xs text-[var(--accent)] hover:underline">
          {showTables ? 'Hide raw tables' : 'Show raw tables (same columns as `rein providers`)'}
        </button>
        {showTables && (
          <div className="mt-3 overflow-x-auto rounded-xl border border-[var(--border)]">
            <table className="w-full text-left text-xs">
              <thead className="border-b border-[var(--border)] bg-[var(--bg-secondary)] uppercase text-[var(--text-muted)]">
                <tr><th className="px-3 py-2">surface</th><th className="px-3 py-2">role</th><th className="px-3 py-2">provider:model</th><th className="px-3 py-2">source</th><th className="px-3 py-2">key</th><th className="px-3 py-2">status</th></tr>
              </thead>
              <tbody>
                {report.surfaces.map((s) => (
                  <tr key={s.surface} className="border-b border-[var(--border)] font-mono">
                    <td className="px-3 py-1.5">{s.surface}</td>
                    <td className="px-3 py-1.5">{ROLE_COPY[s.role as keyof typeof ROLE_COPY]?.title ?? s.role}</td>
                    <td className="px-3 py-1.5">{s.provider === 'none' ? '—' : s.model ? `${s.provider}:${s.model}` : s.provider}</td>
                    <td className="px-3 py-1.5">{s.source_table}</td>
                    <td className="px-3 py-1.5">{s.credential ? credentialName(s.credential) : ''}</td>
                    <td className="px-3 py-1.5">{s.status}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
