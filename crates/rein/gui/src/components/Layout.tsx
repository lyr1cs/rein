import { useState } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { useDoctor, useStats } from '../hooks/useApi';

const NAV_ITEMS = [
  { path: '/', icon: '\u{1F3E0}', label: 'Dashboard' },
  { path: '/brain', icon: '\u{1F9E0}', label: 'Brain View' },
  { path: '/memories', icon: '\u{1F50D}', label: 'Memories' },
  { path: '/adaptive', icon: '\u{1F4CA}', label: 'Adaptive' },
  { path: '/graph', icon: '\u{1F578}️', label: 'Graph' },
  { path: '/timeline', icon: '⏱️', label: 'Timeline' },
  { path: '/artifacts', icon: '\u{1F4C4}', label: 'Artifacts' },
  { path: '/provenance', icon: '\u{1F50F}', label: 'Provenance' },
  { path: '/synthesis-lab', icon: '\u{1F9EA}', label: 'Synthesis Lab' },
];

export default function Layout() {
  const [expanded, setExpanded] = useState(false);
  const location = useLocation();
  const { data: stats } = useStats();
  const { data: doctor } = useDoctor();

  const doctorDotClass = doctor?.status === 'healthy'
    ? 'bg-[var(--success)]'
    : doctor?.status === 'degraded'
      ? 'bg-[var(--warm)]'
      : doctor?.status === 'unhealthy'
        ? 'bg-[var(--hot)]'
        : 'bg-[var(--text-muted)]';

  const doctorTextClass = doctor?.status === 'healthy'
    ? 'text-[var(--success)]'
    : doctor?.status === 'degraded'
      ? 'text-[var(--warm)]'
      : doctor?.status === 'unhealthy'
        ? 'text-[var(--hot)]'
        : 'text-[var(--text-muted)]';

  return (
    <div className="flex h-screen overflow-hidden">
      {/* Sidebar */}
      <aside
        className="flex flex-col items-center border-r border-[var(--border)] bg-[var(--bg-primary)] transition-all duration-200"
        style={{ width: expanded ? 200 : 48 }}
        onMouseEnter={() => setExpanded(true)}
        onMouseLeave={() => setExpanded(false)}
      >
        <div className="flex flex-col items-center gap-2 py-3 w-full">
          {NAV_ITEMS.map((item) => {
            const active = location.pathname === item.path;
            return (
              <Link
                key={item.path}
                to={item.path}
                aria-label={item.label}
                title={item.label}
                aria-current={active ? 'page' : undefined}
                className={`flex items-center gap-3 rounded-lg transition-colors w-full px-2 py-2 ${
                  active
                    ? 'bg-[var(--accent)]/20 shadow-[0_0_12px_var(--accent)/30]'
                    : 'hover:bg-[var(--bg-secondary)]'
                }`}
              >
                {/* L3 (v0.26 cleanup): the emoji icon is decorative — `aria-hidden`
                    keeps screen readers from announcing it twice (once via the
                    aria-label on the parent Link, once via emoji-as-text). The
                    visible label slides in on hover for sighted users. */}
                <span
                  aria-hidden="true"
                  className="text-base w-8 h-8 flex items-center justify-center flex-shrink-0"
                >
                  {item.icon}
                </span>
                {expanded && (
                  <span className="text-sm text-[var(--text-secondary)] whitespace-nowrap overflow-hidden">
                    {item.label}
                  </span>
                )}
              </Link>
            );
          })}
        </div>
        <div className="mt-auto pb-3 px-2 w-full">
          <Link
            to="/settings"
            aria-label="Settings"
            title="Settings"
            aria-current={location.pathname === '/settings' ? 'page' : undefined}
            className="flex items-center gap-3 rounded-lg hover:bg-[var(--bg-secondary)] px-2 py-2 w-full"
          >
            <span aria-hidden="true" className="text-base w-8 h-8 flex items-center justify-center flex-shrink-0">{'⚙️'}</span>
            {expanded && <span className="text-sm text-[var(--text-muted)]">Settings</span>}
          </Link>
        </div>
      </aside>

      {/* Main content */}
      <div className="flex-1 flex flex-col overflow-hidden">
        {/* Vitals bar */}
        <header className="h-10 flex items-center px-4 gap-5 border-b border-[var(--border)] bg-[var(--bg-primary)]/80 backdrop-blur-sm text-xs">
          {stats && (
            <>
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--new)]" />
                <span className="text-[var(--new)] font-mono">{stats.total_memories.toLocaleString()}</span>
                <span className="text-[var(--text-muted)]">memories</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--warm)]" />
                <span className="text-[var(--warm)] font-mono">{stats.concept_count}</span>
                <span className="text-[var(--text-muted)]">concepts</span>
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-1.5 h-1.5 rounded-full bg-[var(--success)]" />
                <span className="text-[var(--success)] font-mono">{stats.memoir_count}</span>
                <span className="text-[var(--text-muted)]">memoirs</span>
              </span>
              {doctor && (
                <Link
                  to="/settings"
                  className="ml-auto flex items-center gap-1.5 rounded-full border border-[var(--border)] px-2.5 py-1 transition-colors hover:bg-[var(--bg-secondary)]"
                >
                  <span className={`h-1.5 w-1.5 rounded-full ${doctorDotClass}`} />
                  <span className={`font-mono ${doctorTextClass}`}>{doctor.status}</span>
                  <span className="text-[var(--text-muted)]">doctor</span>
                </Link>
              )}
            </>
          )}
        </header>

        {/* Page content */}
        <main className="flex-1 overflow-auto">
          <Outlet />
        </main>
      </div>
    </div>
  );
}
