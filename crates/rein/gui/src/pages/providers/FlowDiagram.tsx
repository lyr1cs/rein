import { useEffect, useRef, useState } from 'react';
import type { ProviderStatusReport } from '../../api/types';
import { ROLE_COPY } from './copy';
import type { RoleStatus } from './providersModel';
import { credentialName, flowLayout, groupByRole, representative, roleStatus } from './providersModel';

const STATUS_STROKE: Record<RoleStatus, string> = {
  ready: 'var(--success)',
  missing: 'var(--hot)',
  disabled: 'var(--text-muted)',
};

interface Props {
  report: ProviderStatusReport;
  onSelectRole: (role: string) => void;
}

export default function FlowDiagram({ report, onSelectRole }: Props) {
  const ref = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(960);
  useEffect(() => {
    const el = ref.current;
    if (!el) return;
    const ro = new ResizeObserver((entries) => {
      const w = entries[0]?.contentRect.width;
      if (w) setWidth(Math.floor(w));
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);
  const layout = flowLayout(width);
  const groups = groupByRole(report);
  const byRole = Object.fromEntries(groups.map((g) => [g.role, g]));
  const centre = (id: string) => {
    const n = layout.nodes.find((x) => x.id === id)!;
    return { x: n.x + n.w / 2, y: n.y + n.h / 2, n };
  };
  const path = (from: string, to: string) => {
    const a = centre(from), b = centre(to);
    const vertical = width < 768 || Math.abs(a.x - b.x) < 8;
    if (vertical) {
      const y1 = a.y < b.y ? a.n.y + a.n.h : a.n.y;
      const y2 = a.y < b.y ? b.n.y : b.n.y + b.n.h;
      return `M ${a.x} ${y1} C ${a.x} ${(y1 + y2) / 2}, ${b.x} ${(y1 + y2) / 2}, ${b.x} ${y2}`;
    }
    const x1 = a.x < b.x ? a.n.x + a.n.w : a.n.x;
    const x2 = a.x < b.x ? b.n.x : b.n.x + b.n.w;
    return `M ${x1} ${a.y} C ${(x1 + x2) / 2} ${a.y}, ${(x1 + x2) / 2} ${b.y}, ${x2} ${b.y}`;
  };
  return (
    <div ref={ref} className="w-full overflow-x-auto">
      <svg viewBox={`0 0 ${layout.width} ${layout.height + 8}`} width="100%" role="img" aria-label="How rein uses AI models">
        <defs>
          <marker id="arrow" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
            <path d="M 0 0 L 10 5 L 0 10 z" fill="var(--text-muted)" />
          </marker>
        </defs>
        {layout.edges.map((e) => (
          <g key={`${e.from}-${e.to}`}>
            <path d={path(e.from, e.to)} fill="none" stroke="var(--text-muted)" strokeWidth="1.5" strokeDasharray={e.dashed ? '5 4' : undefined} markerEnd="url(#arrow)" opacity={e.dashed ? 0.6 : 0.9} />
            {e.label && (
              <text x={(centre(e.from).x + centre(e.to).x) / 2} y={(centre(e.from).y + centre(e.to).y) / 2 - 6} textAnchor="middle" fontSize="10" fill="var(--text-muted)">
                {e.label}
              </text>
            )}
          </g>
        ))}
        {layout.nodes.map((n) => {
          if (n.kind === 'data') {
            const extra = n.id === 'store' && report.vector_index ? `${report.vector_index.rows.toLocaleString()} memories · index ${report.vector_index.matches ? '✓' : '✗'}` : '';
            return (
              <g key={n.id}>
                <rect x={n.x} y={n.y} width={n.w} height={n.h} rx="6" fill="var(--bg-secondary)" stroke="var(--border)" />
                <text x={n.x + n.w / 2} y={n.y + (extra ? 24 : 33)} textAnchor="middle" fontSize="12" fill="var(--text-secondary)">{n.label}</text>
                {extra && <text x={n.x + n.w / 2} y={n.y + 42} textAnchor="middle" fontSize="10" fill="var(--text-muted)">{extra}</text>}
              </g>
            );
          }
          const group = byRole[n.id];
          const status = group ? roleStatus(group.surfaces) : 'disabled';
          const rep = group ? representative(group.surfaces) : undefined;
          const copy = ROLE_COPY[n.id as keyof typeof ROLE_COPY];
          const model = rep && rep.provider !== 'none' ? (rep.model ? `${rep.provider}:${rep.model}` : rep.provider) : 'off';
          const keyText = rep?.credential ? credentialName(rep.credential) : rep && rep.status !== 'disabled' ? 'local' : '';
          const cred = rep?.credential ? report.credentials.find((c) => c.id === rep.credential) : undefined;
          const keyColor = !rep?.credential ? 'var(--text-muted)' : cred?.present ? 'var(--success)' : 'var(--hot)';
          return (
            <g key={n.id} tabIndex={0} role="button" aria-label={`${copy.title}: ${status}`} onClick={() => onSelectRole(n.id)} onKeyDown={(ev) => { if (ev.key === 'Enter' || ev.key === ' ') onSelectRole(n.id); }} style={{ cursor: 'pointer' }} opacity={status === 'disabled' ? 0.55 : 1}>
              <title>{group?.surfaces.map((x) => `${x.surface}: ${x.provider}${x.model ? ':' + x.model : ''} (${x.status})`).join('\n')}</title>
              <rect x={n.x} y={n.y} width={n.w} height={n.h} rx="10" fill="var(--bg-secondary)" stroke={STATUS_STROKE[status]} strokeWidth="2" />
              <text x={n.x + 12} y={n.y + 24} fontSize="13" fontWeight="600" fill="var(--text-primary)">{copy.icon} {copy.title}</text>
              <text x={n.x + 12} y={n.y + 46} fontSize="11" fill="var(--text-secondary)">
                {model.length > 26 ? model.slice(0, 25) + '…' : model}
              </text>
              {keyText && (
                <g>
                  <circle cx={n.x + 17} cy={n.y + 68} r="4" fill={keyColor} />
                  <text x={n.x + 26} y={n.y + 72} fontSize="10" fill="var(--text-muted)">{keyText}</text>
                </g>
              )}
            </g>
          );
        })}
      </svg>
    </div>
  );
}
