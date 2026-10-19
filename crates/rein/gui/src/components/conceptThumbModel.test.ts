import { describe, expect, it } from 'vitest';
import { beginThumb, canCastThumb, idleThumbVote, settleThumb } from './conceptThumbModel';

describe('conceptThumbModel', () => {
  it('does not latch a vote the server refused and keeps retry open', () => {
    const inFlight = beginThumb(idleThumbVote);
    expect(inFlight.pending).toBe(true);
    expect(canCastThumb(inFlight)).toBe(false);

    const refused = settleThumb(true, { ok: false, reason: 'HTTP 403' });
    expect(refused.state).toBeNull();
    expect(refused.pending).toBe(false);
    expect(refused.error).toBe('HTTP 403');
    expect(canCastThumb(refused)).toBe(true);
  });

  it('latches only after the server accepted the vote', () => {
    const accepted = settleThumb(false, { ok: true });
    expect(accepted).toEqual({ state: 'down', pending: false, error: null });
    expect(canCastThumb(accepted)).toBe(false);
  });

  it('clears a previous failure when a retry starts', () => {
    const refused = settleThumb(true, { ok: false, reason: 'offline' });
    const retry = beginThumb(refused);
    expect(retry.error).toBeNull();
    expect(settleThumb(true, { ok: true })).toEqual({ state: 'up', pending: false, error: null });
  });
});
