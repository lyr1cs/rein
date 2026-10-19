import type { ProviderStatusReport, ProviderSurface } from '../../api/types';
import { ROLE_ORDER } from './copy';

export type RoleStatus = 'ready' | 'missing' | 'disabled';

export interface RoleGroup {
  role: (typeof ROLE_ORDER)[number];
  surfaces: ProviderSurface[];
}

export function groupByRole(report: ProviderStatusReport): RoleGroup[] {
  return ROLE_ORDER.map((role) => ({
    role,
    surfaces: report.surfaces.filter((s) => s.role === role),
  }));
}

export function roleStatus(surfaces: ProviderSurface[]): RoleStatus {
  const enabled = surfaces.filter((s) => s.enabled && s.status !== 'disabled');
  if (enabled.length === 0) return 'disabled';
  if (enabled.some((s) => s.status === 'missing_credential')) return 'missing';
  return 'ready';
}

export function credentialName(id: string): string {
  const i = id.indexOf(':');
  return i === -1 ? id : id.slice(i + 1);
}

export function summarize(report: ProviderStatusReport) {
  const groups = groupByRole(report);
  const statuses = groups.map((g) => roleStatus(g.surfaces));
  const keyCreds = report.credentials.filter((c) => c.kind !== 'endpoint');
  return {
    keys: keyCreds.length,
    presentKeys: keyCreds.filter((c) => c.present).length,
    readyRoles: statuses.filter((s) => s === 'ready').length,
    missingRoles: statuses.filter((s) => s === 'missing').length,
    disabledRoles: statuses.filter((s) => s === 'disabled').length,
  };
}

/** The surface whose provider/model represents the role on the diagram. */
export function representative(surfaces: ProviderSurface[]): ProviderSurface | undefined {
  return surfaces.find((s) => s.enabled && s.status !== 'disabled') ?? surfaces[0];
}

export interface FlowNode {
  id: string;
  kind: 'role' | 'data';
  label: string;
  x: number;
  y: number;
  w: number;
  h: number;
}

export interface FlowEdge {
  from: string;
  to: string;
  dashed?: boolean;
  label?: string;
}

export interface FlowLayout {
  width: number;
  height: number;
  nodes: FlowNode[];
  edges: FlowEdge[];
}

const ROLE_W = 180;
const ROLE_H = 92;
const DATA_W = 150;
const DATA_H = 56;
const GAP = 36;

/**
 * Writing lane:  [Sessions] → (write) → [Memory store] → (embed)
 * Reading lane:  [Question] → (search) → [Recall] → (answer) → [Answer]
 *                                           ▲                    ↓
 *                 (cloud) ┄▶ [Recall]       └── adaptive ◀── (judge)
 * Below 768px every node stacks in one column in reading order.
 */
export function flowLayout(width: number): FlowLayout {
  const nodes: FlowNode[] = [];
  const edges: FlowEdge[] = [];
  const add = (id: string, kind: FlowNode['kind'], label: string, x: number, y: number) => {
    const w = kind === 'role' ? ROLE_W : DATA_W;
    const h = kind === 'role' ? ROLE_H : DATA_H;
    nodes.push({ id, kind, label, x, y, w, h });
  };
  if (width < 768) {
    const order: Array<[string, FlowNode['kind'], string]> = [
      ['sessions', 'data', 'Sessions & tool output'],
      ['write', 'role', 'Write memories'],
      ['store', 'data', 'Memory store'],
      ['embed', 'role', 'Vectorize memories'],
      ['question', 'data', 'Your question'],
      ['search', 'role', 'Sharpen search'],
      ['recall', 'data', 'Recall'],
      ['answer', 'role', 'Answer & summarize'],
      ['out', 'data', 'Answer'],
      ['judge', 'role', 'Grade & learn'],
      ['cloud', 'role', 'Cloud cross-check'],
    ];
    let y = 0;
    for (const [id, kind, label] of order) {
      add(id, kind, label, 0, y);
      y += (kind === 'role' ? ROLE_H : DATA_H) + 24;
    }
    for (let i = 0; i + 1 < order.length; i++) {
      if (order[i][0] === 'out' || order[i][0] === 'judge') continue;
      edges.push({ from: order[i][0], to: order[i + 1][0] });
    }
    edges.push({ from: 'out', to: 'judge' });
    edges.push({ from: 'judge', to: 'search', label: 'adaptive weights' });
    edges.push({ from: 'cloud', to: 'recall', dashed: true });
    return { width: Math.max(width, ROLE_W), height: y, nodes, edges };
  }
  const cols = 5;
  const usable = Math.max(width, cols * ROLE_W + (cols - 1) * GAP);
  const step = (usable - ROLE_W) / (cols - 1);
  const cx = (i: number, w: number) => Math.round(i * step + (ROLE_W - w) / 2);
  const lane1 = 0;
  const lane2 = ROLE_H + 72;
  const lane3 = lane2 + ROLE_H + 60;
  add('sessions', 'data', 'Sessions & tool output', cx(0, DATA_W), lane1 + (ROLE_H - DATA_H) / 2);
  add('write', 'role', 'Write memories', cx(1, ROLE_W), lane1);
  add('store', 'data', 'Memory store', cx(2, DATA_W), lane1 + (ROLE_H - DATA_H) / 2);
  add('embed', 'role', 'Vectorize memories', cx(3, ROLE_W), lane1);
  add('question', 'data', 'Your question', cx(0, DATA_W), lane2 + (ROLE_H - DATA_H) / 2);
  add('search', 'role', 'Sharpen search', cx(1, ROLE_W), lane2);
  add('recall', 'data', 'Recall', cx(2, DATA_W), lane2 + (ROLE_H - DATA_H) / 2);
  add('answer', 'role', 'Answer & summarize', cx(3, ROLE_W), lane2);
  add('out', 'data', 'Answer', cx(4, DATA_W), lane2 + (ROLE_H - DATA_H) / 2);
  add('cloud', 'role', 'Cloud cross-check', cx(1, ROLE_W), lane3);
  add('judge', 'role', 'Grade & learn', cx(3, ROLE_W), lane3);
  edges.push({ from: 'sessions', to: 'write' }, { from: 'write', to: 'store' }, { from: 'store', to: 'embed' });
  edges.push({ from: 'question', to: 'search' }, { from: 'search', to: 'recall' }, { from: 'recall', to: 'answer' }, { from: 'answer', to: 'out' });
  edges.push({ from: 'out', to: 'judge' }, { from: 'judge', to: 'search', label: 'adaptive weights' });
  edges.push({ from: 'cloud', to: 'recall', dashed: true });
  return { width: usable, height: lane3 + ROLE_H, nodes, edges };
}
