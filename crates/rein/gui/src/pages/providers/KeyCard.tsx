import type { ProviderCredential, ProviderStatusReport } from '../../api/types';
import EyeToggle from '../../components/EyeToggle';
import { ROLE_COPY, VENDOR_COPY } from './copy';
import { credentialName } from './providersModel';

const PROBE_BADGE: Record<string, string> = {
  ok: 'text-[var(--success)] border-[var(--success)]/30',
  unauthorized: 'text-[var(--hot)] border-[var(--hot)]/30',
  unreachable: 'text-[var(--text-muted)] border-[var(--border)]',
  dimension_mismatch: 'text-[var(--warm)] border-[var(--warm)]/30',
  error: 'text-[var(--warm)] border-[var(--warm)]/30',
};

interface Props {
  credential: ProviderCredential;
  report: ProviderStatusReport;
  revealed: string | null;
  revealError: string | null;
  onToggleReveal: () => void;
  highlighted: boolean;
  onSelectRole: (role: string) => void;
}

export default function KeyCard({ credential: c, report, revealed, revealError, onToggleReveal, highlighted, onSelectRole }: Props) {
  const name = credentialName(c.id);
  const surfaces = report.surfaces.filter((s) => s.credential === c.id);
  const roles = Array.from(new Set(surfaces.map((s) => s.role)));
  const vendor = surfaces[0]?.provider ?? '';
  const vcopy = VENDOR_COPY[vendor] ?? { label: `OpenAI-compatible endpoint: ${surfaces[0]?.endpoint ?? ''}` };
  const source = c.kind === 'env' ? 'process environment' : c.kind === 'secrets_file' ? '~/.rein/secrets.env' : c.kind === 'config_file' ? 'config.toml' : 'endpoint';
  return (
    <div id={`key-${name}`} className={`rounded-xl border bg-[var(--bg-secondary)] p-5 space-y-3 ${highlighted ? 'border-[var(--accent)]' : c.kind === 'config_file' ? 'border-[var(--warm)]/40' : 'border-[var(--border)]'}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="font-mono text-sm text-[var(--text-primary)]">{name}</div>
        <span className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${c.present ? 'text-[var(--success)] border-[var(--success)]/20 bg-[var(--success)]/10' : 'text-[var(--hot)] border-[var(--hot)]/20 bg-[var(--hot)]/10'}`}>
          {c.present ? 'present' : 'missing'}
        </span>
      </div>
      <div className="text-xs text-[var(--text-secondary)]">
        {vcopy.label}
        {vcopy.keyUrl && (
          <a href={vcopy.keyUrl} target="_blank" rel="noopener noreferrer" className="ml-2 text-[var(--accent)] hover:underline">Get a key ↗</a>
        )}
      </div>
      <div className="flex flex-wrap items-center gap-1 text-xs">
        <span className="text-[var(--text-muted)]">Powers:</span>
        {roles.map((r) => (
          <button key={r} type="button" onClick={() => onSelectRole(r)} className="rounded border border-[var(--border)] px-1.5 py-0.5 text-[var(--text-secondary)] hover:border-[var(--accent)]">
            {ROLE_COPY[r as keyof typeof ROLE_COPY]?.icon} {ROLE_COPY[r as keyof typeof ROLE_COPY]?.title ?? r}
          </button>
        ))}
      </div>
      {c.present ? (
        <div className="flex items-center gap-2">
          <code className="flex-1 overflow-x-auto rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 text-xs text-[var(--text-primary)]">
            {revealed ?? c.masked ?? '••••'}
          </code>
          <EyeToggle shown={revealed !== null} onToggle={onToggleReveal} label={name} />
        </div>
      ) : (
        <pre className="rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-1.5 text-[11px] text-[var(--text-secondary)]">rein keys set {name}</pre>
      )}
      {revealError && <div className="rounded-lg border border-[var(--hot)]/20 bg-[var(--hot)]/10 px-3 py-2 text-xs text-[var(--hot)]">{revealError}</div>}
      {c.warning && <div className="rounded-lg border border-[var(--warm)]/20 bg-[var(--warm)]/10 px-3 py-2 text-xs text-[var(--warm)]">{c.warning}</div>}
      <div className="text-[11px] text-[var(--text-muted)]">Where rein reads it: {source}</div>
      {c.probe && (
        <div className={`inline-flex items-center gap-2 rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${PROBE_BADGE[c.probe.status] ?? PROBE_BADGE.error}`}>
          probe {c.probe.status}
          {c.probe.http_status !== null && <span>({c.probe.http_status})</span>}
          <span className="normal-case">{c.probe.latency_ms} ms</span>
          {c.probe.detail && <span className="normal-case">{c.probe.detail}</span>}
        </div>
      )}
    </div>
  );
}
