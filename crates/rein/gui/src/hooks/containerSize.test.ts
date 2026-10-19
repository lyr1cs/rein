import { describe, expect, it } from 'vitest';
import { sizeFromEntries } from './containerSize';

describe('containerSize', () => {
  it('adopts the last positive resize entry and ignores collapsed boxes', () => {
    const current = { width: 800, height: 600 };
    const next = sizeFromEntries(
      [
        { contentRect: { width: 900.9, height: 500.1 } },
        { contentRect: { width: 0, height: 0 } },
      ],
      current,
    );
    expect(next).toEqual({ width: 900, height: 500 });
    expect(sizeFromEntries([{ contentRect: { width: 0, height: 300 } }], current)).toBe(current);
    expect(sizeFromEntries([], current)).toBe(current);
  });
});
