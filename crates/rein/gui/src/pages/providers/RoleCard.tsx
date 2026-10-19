import { useState } from 'react';
import type { ProviderStatusReport, ProviderSurface } from '../../api/types';
import { ROLE_COPY, STATUS_LABEL } from './copy';
import type { RoleStatus } from './providersModel';
import { credentialName, representative } from './providersModel';

const BADGE: Record<RoleStatus, string> = {
  ready: 'text-[var(--success)] bg-[var(--success)]/10 border-[var(--success)]/20',
  missing: 'text-[var(--hot)] bg-[var(--hot)]/10 border-[var(--hot)]/20',
  disabled: 'text-[var(--text-muted)] bg-[var(--bg-tertiary)] border-[var(--border)]',
};

interface Props {
  role: keyof typeof ROLE_COPY;
  surfaces: ProviderSurface[];
  status: RoleStatus;
  report: ProviderStatusReport;
  highlighted: boolean;
  onSelectKey: (credentialId: string) => void;
}

function fill(template: string, rep: ProviderSurface | undefined, dims: string) {
  const key = rep?.credential ? credentialName(rep.credential) : '{key}';
  const vendor = rep?.provider && rep.provider !== 'none' ? rep.provider : '{vendor}';
  return template
    .replace(/\{vendor\}/g, vendor)
    .replace(/\{table\}/g, vendor === 'google' ? 'google' : 'openai')
    .replace(/\{model\}/g, rep?.model || '{model}')
    .replace(/\{dims\}/g, dims)
    .replace(/\{key\}/g, key);
}

/** Dimensions of the configured embedding model, read from the `provider:model:dims` stamp. */
function configuredDims(report: ProviderStatusReport): string {
  const tail = report.vector_index?.configured.split(':').pop();
  return tail && /^\d+$/.test(tail) ? tail : '{dims}';
}

export default function RoleCard({ role, surfaces, status, report, highlighted, onSelectKey }: Props) {
  const copy = ROLE_COPY[role];
  const rep = representative(surfaces);
  const dims = configuredDims(report);
  const [open, setOpen] = useState(false);
  return (
    <div id={`role-${role}`} className={`rounded-xl border bg-[var(--bg-secondary)] p-5 space-y-3 ${highlighted ? 'border-[var(--accent)]' : 'border-[var(--border)]'}`}>
      <div className="flex items-center justify-between gap-2">
        <div className="text-sm font-medium text-[var(--text-primary)]">{copy.icon} {copy.title}</div>
        <span className={`rounded-full border px-2 py-0.5 text-[10px] uppercase tracking-wider ${BADGE[status]}`}>{STATUS_LABEL[status]}</span>
      </div>
      <p className="text-xs text-[var(--text-secondary)]">{copy.what}</p>
      <div className="grid grid-cols-1 gap-2 text-xs md:grid-cols-2">
        <div><span className="text-[var(--text-muted)]">Runs: </span><span className="text-[var(--text-secondary)]">{copy.runs}</span></div>
        <div><span className="text-[var(--text-muted)]">Without it: </span><span className="text-[var(--text-secondary)]">{copy.without}</span></div>
      </div>
      <div className="space-y-1">
        <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">Now wired to</div>
        {surfaces.map((s) => {
          const cred = s.credential ? report.credentials.find((c) => c.id === s.credential) : undefined;
          const off = !s.enabled || s.status === 'disabled';
          return (
            <div key={s.surface} className={`flex flex-wrap items-center gap-2 font-mono text-[11px] ${off ? 'text-[var(--text-muted)]' : 'text-[var(--text-secondary)]'}`}>
              <span>{s.surface}</span>
              <span>·</span>
              <span>{s.provider === 'none' ? 'off' : s.model ? `${s.provider}:${s.model}` : s.provider}</span>
              <span className="text-[var(--text-muted)]">{s.source_table}</span>
              {s.credential && (
                <button type="button" onClick={() => onSelectKey(s.credential!)} className={`rounded border px-1.5 py-0.5 ${cred?.present ? 'border-[var(--success)]/30 text-[var(--success)]' : 'border-[var(--hot)]/30 text-[var(--hot)]'}`}>
                  {credentialName(s.credential)}
                </button>
              )}
              {!s.credential && !off && <span className="rounded border border-[var(--border)] px-1.5 py-0.5 text-[var(--text-muted)]">local</span>}
            </div>
          );
        })}
      </div>
      <button type="button" onClick={() => setOpen(!open)} className="text-xs text-[var(--accent)] hover:underline">
        {open ? 'Hide' : 'How to change'}
      </button>
      {open && (
        <div className="space-y-2">
          <pre className="overflow-x-auto rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2 text-[11px] text-[var(--text-secondary)]">{copy.cli.map((c) => fill(c, rep, dims)).join('\n')}</pre>
          <div className="text-[10px] uppercase tracking-wider text-[var(--text-muted)]">equivalent config.toml</div>
          <pre className="overflow-x-auto rounded-md border border-[var(--border)] bg-[var(--bg-primary)] px-3 py-2 text-[11px] text-[var(--text-secondary)]">{fill(copy.toml, rep, dims)}</pre>
        </div>
      )}
    </div>
  );
}
