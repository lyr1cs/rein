import { useState } from 'react';
import { Link, Outlet, useLocation } from 'react-router-dom';
import { useStats } from '../hooks/useApi';

const NAV_ITEMS = [
  { path: '/', icon: '\u{1F3E0}', label: 'Dashboard' },
  { path: '/brain', icon: '\u{1F9E0}', label: 'Brain View' },
  { path: '/memories', icon: '\u{1F50D}', label: 'Memories' },
  { path: '/adaptive', icon: '\u{1F4CA}', label: 'Adaptive' },
  { path: '/graph', icon: '\u{1F578}\uFE0F', label: 'Graph' },
  { path: '/timeline', icon: '\u23F1\uFE0F', label: 'Timeline' },
  { path: '/artifacts', icon: '\u{1F4C4}', label: 'Artifacts' },
];

export default function Layout() {
  const [expanded, setExpanded] = useState(false);
  const location = useLocation();
  const { data: stats } = useStats();

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
                className={`flex items-center gap-3 rounded-lg transition-colors w-full px-2 py-2 ${
                  active
                    ? 'bg-[var(--accent)]/20 shadow-[0_0_12px_var(--accent)/30]'
                    : 'hover:bg-[var(--bg-secondary)]'
                }`}
              >
                <span className="text-base w-8 h-8 flex items-center justify-center flex-shrink-0">
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
            className="flex items-center gap-3 rounded-lg hover:bg-[var(--bg-secondary)] px-2 py-2 w-full"
          >
            <span className="text-base w-8 h-8 flex items-center justify-center flex-shrink-0">{'\u2699\uFE0F'}</span>
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
