import { describe, expect, it } from 'vitest';
import type { ProviderStatusReport } from '../../api/types';
import { credentialName, flowLayout, groupByRole, roleStatus, summarize } from './providersModel';
import { ROLE_ORDER } from './copy';

const s = (surface: string, role: string, enabled: boolean, provider: string, model: string, status: string, credential: string | null, source_table: string): ProviderStatusReport['surfaces'][number] => ({
  surface, role, enabled, provider, model, endpoint: 'https://x', source_level: 'section', source_table, credential, status,
});

const report: ProviderStatusReport = {
  surfaces: [
    s('embedding', 'embed', true, 'google', 'gemini-embedding-001', 'ready', 'env:GEMINI_API_KEY', '[embedding.google]'),
    s('extract', 'write', true, 'google', 'gemini-3.5-flash-lite', 'ready', 'env:GEMINI_API_KEY', '[extract.google]'),
    s('extract.async_memory', 'write', true, 'google', 'gemini-3.5-flash-lite', 'ready', 'env:GEMINI_API_KEY', '[extract.google]'),
    s('extract.intelligent_merge', 'write', false, 'google', 'x', 'disabled', 'env:GEMINI_API_KEY', '[intelligent_merge.google]'),
    s('query_expansion', 'search', true, 'google', 'gemini-3.5-flash-lite', 'ready', 'env:GEMINI_API_KEY', '[query_expansion.google]'),
    s('search.llm_reranker', 'search', false, 'none', '', 'disabled', null, 'none'),
    s('ars.recall_synthesis', 'answer', true, 'google', 'gemini-3.6-flash', 'ready', 'env:GEMINI_API_KEY', '[llm.google]'),
    s('ars.concept_summary', 'answer', true, 'google', 'gemini-3.6-flash', 'ready', 'env:GEMINI_API_KEY', '[llm.google]'),
    s('ars.cold_archive', 'answer', true, 'google', 'gemini-3.6-flash', 'ready', 'env:GEMINI_API_KEY', '[llm.google]'),
    s('resummerize', 'write', false, 'google', 'x', 'disabled', 'env:GEMINI_API_KEY', '[llm.google]'),
    s('ars.llm_judge', 'judge', true, 'openrouter', 'anthropic/claude-sonnet-4.5', 'missing_credential', 'env:OPENROUTER_API_KEY', '[ars.llm_judge.openai]'),
    s('ars.llm_judge.nightly_cron', 'judge', true, 'openrouter', 'anthropic/claude-sonnet-4.5', 'missing_credential', 'env:OPENROUTER_API_KEY', '[ars.llm_judge.openai]'),
    s('supermemory', 'cloud', false, 'supermemory', '', 'disabled', null, '[sync]'),
  ],
  credentials: [
    { id: 'env:GEMINI_API_KEY', kind: 'secrets_file', present: true, masked: 'AIza••••3f2a', used_by: ['embedding', 'extract'], warning: null, probe: null },
    { id: 'env:OPENROUTER_API_KEY', kind: 'env', present: false, masked: null, used_by: ['ars.llm_judge'], warning: null, probe: null },
  ],
  vector_index: { stamped: 'google:gemini-embedding-001:3072', configured: 'google:gemini-embedding-001:3072', matches: true, rows: 11012 },
  probed_at: null,
};

describe('groupByRole', () => {
  it('yields every role in ROLE_ORDER with its surfaces in report order', () => {
    const groups = groupByRole(report);
    expect(groups.map((g) => g.role)).toEqual(ROLE_ORDER);
    expect(groups.find((g) => g.role === 'write')!.surfaces.map((x) => x.surface)).toEqual([
      'extract', 'extract.async_memory', 'extract.intelligent_merge', 'resummerize',
    ]);
  });
});

describe('roleStatus', () => {
  it('is missing when any enabled surface misses a credential, disabled when all are off, else ready', () => {
    const g = groupByRole(report);
    const byRole = Object.fromEntries(g.map((x) => [x.role, roleStatus(x.surfaces)]));
    expect(byRole.embed).toBe('ready');
    expect(byRole.write).toBe('ready');
    expect(byRole.judge).toBe('missing');
    expect(byRole.cloud).toBe('disabled');
  });
});

describe('summarize', () => {
  it('counts keys, ready roles and missing roles', () => {
    expect(summarize(report)).toEqual({ keys: 2, presentKeys: 1, readyRoles: 4, missingRoles: 1, disabledRoles: 1 });
  });
});

describe('credentialName', () => {
  it('strips the kind prefix', () => {
    expect(credentialName('env:GEMINI_API_KEY')).toBe('GEMINI_API_KEY');
    expect(credentialName('config:extract.google.api_key')).toBe('extract.google.api_key');
    expect(credentialName('endpoint:http://x')).toBe('http://x');
  });
});

describe('flowLayout', () => {
  it('places six role nodes and data boxes without overlap inside the viewBox', () => {
    const layout = flowLayout(960);
    expect(layout.nodes.filter((n) => n.kind === 'role').map((n) => n.id).sort()).toEqual([...ROLE_ORDER].sort());
    for (const n of layout.nodes) {
      expect(n.x).toBeGreaterThanOrEqual(0);
      expect(n.x + n.w).toBeLessThanOrEqual(layout.width);
      expect(n.y).toBeGreaterThanOrEqual(0);
      expect(n.y + n.h).toBeLessThanOrEqual(layout.height);
    }
    const boxes = layout.nodes;
    for (let i = 0; i < boxes.length; i++) {
      for (let j = i + 1; j < boxes.length; j++) {
        const a = boxes[i], b = boxes[j];
        const overlap = a.x < b.x + b.w && b.x < a.x + a.w && a.y < b.y + b.h && b.y < a.y + a.h;
        expect(overlap, `${a.id} overlaps ${b.id}`).toBe(false);
      }
    }
    for (const e of layout.edges) {
      expect(layout.nodes.some((n) => n.id === e.from)).toBe(true);
      expect(layout.nodes.some((n) => n.id === e.to)).toBe(true);
    }
  });

  it('stacks vertically below 768px', () => {
    const narrow = flowLayout(390);
    const xs = new Set(narrow.nodes.map((n) => n.x));
    expect(xs.size).toBe(1);
  });
});
