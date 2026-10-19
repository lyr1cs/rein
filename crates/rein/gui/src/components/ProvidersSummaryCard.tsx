import { Link } from 'react-router-dom';
import { useProviders } from '../hooks/useApi';
import { credentialName, summarize } from '../pages/providers/providersModel';

export default function ProvidersSummaryCard() {
  const { data } = useProviders(false);
  if (!data) return null;
  const sum = summarize(data);
  const keys = data.credentials.filter((c) => c.kind !== 'endpoint');
  return (
    <div className="mb-8 rounded-xl border border-[var(--border)] bg-[var(--bg-secondary)] p-4">
      <div className="mb-2 flex items-center justify-between">
        <div className="text-xs uppercase tracking-wider text-[var(--text-muted)]">AI Keys</div>
        <Link to="/providers" className="text-xs text-[var(--accent)] hover:underline">Details →</Link>
      </div>
      <div className="mb-2 text-xs text-[var(--text-secondary)]">
        {sum.readyRoles} roles ready · {sum.missingRoles > 0 ? <span className="text-[var(--hot)]">{sum.missingRoles} missing a key</span> : 'no missing keys'} · {sum.disabledRoles} off
      </div>
      <div className="flex flex-wrap gap-2">
        {keys.map((c) => (
          <span key={c.id} className="inline-flex items-center gap-1.5 rounded border border-[var(--border)] px-2 py-0.5 font-mono text-[11px] text-[var(--text-secondary)]">
            <span className="h-2 w-2 rounded-full" style={{ background: c.present ? 'var(--success)' : 'var(--hot)' }} />
            {credentialName(c.id)} {c.masked ?? 'missing'} <span className="text-[var(--text-muted)]">×{c.used_by.length}</span>
          </span>
        ))}
        {keys.length === 0 && <span className="text-xs text-[var(--text-muted)]">no API keys configured</span>}
      </div>
    </div>
  );
}
