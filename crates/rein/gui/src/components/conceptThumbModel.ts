import type { FeedbackPostResult } from '../api/feedback';

/**
 * State machine behind the concept-summary thumb buttons. Kept free of
 * React so the latch rule is unit-testable: a vote is recorded only after
 * the server accepted it. The previous optimistic latch flipped the buttons
 * to "You rated this…" before the POST returned and ignored its result, so
 * a rejected write (network failure, or the 403 every public-auth REST
 * mutation gets) looked recorded while nothing was stored, and the card
 * offered no retry until it remounted.
 */
export type ThumbState = 'up' | 'down' | null;

export interface ThumbVote {
  /** The accepted vote, or `null` while none has been accepted. */
  state: ThumbState;
  /** A POST is in flight; both buttons stay disabled meanwhile. */
  pending: boolean;
  /** Reason the last attempt was refused; cleared on the next attempt. */
  error: string | null;
}

export const idleThumbVote: ThumbVote = { state: null, pending: false, error: null };

/** A click is honoured only when nothing is latched and nothing is in flight. */
export function canCastThumb(vote: ThumbVote): boolean {
  return vote.state === null && !vote.pending;
}

export function beginThumb(vote: ThumbVote): ThumbVote {
  return { ...vote, pending: true, error: null };
}

/** Latch on success; on failure stay unlatched (retry stays possible) and surface the reason. */
export function settleThumb(up: boolean, result: FeedbackPostResult): ThumbVote {
  return result.ok
    ? { state: up ? 'up' : 'down', pending: false, error: null }
    : { state: null, pending: false, error: result.reason };
}
