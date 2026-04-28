/**
 * v0.26 D direction — single helper for posting feedback events to
 * `/api/feedback`. Reuses the shared `apiPost` from `client.ts` so the
 * `x-rein-action: 1` mutation marker + `credentials: same-origin` flow stay
 * consistent across the GUI.
 *
 * **Failure policy**: silent. Feedback is observability — a failed POST
 * must NEVER break the recall surface or surprise the user with a toast.
 * We log to `console.warn` (not `console.error`) so dev-tools surfaces the
 * issue without polluting the JS error budget.
 *
 * **synthesis_id gate (per v0.26 contract §8 invariant 9)**: callers must
 * pass a non-empty `synthesis_id` and `recall_id`. The wrapper guards both
 * defensively so accidental `undefined`/`""` never reaches the wire — a
 * missing id means the synthesis was either skipped or the server hasn't
 * shipped Cap C yet, and posting in either case would corrupt the
 * `synthesis_feedback` consumer's stats.
 */
import { apiPost } from './client';
import type {
  ConceptSummaryInteractionEvent,
  ConceptSummaryInteractionKind,
  ConceptSummaryMetadata,
  FeedbackPayload,
  SynthesisInteractionKind,
  SynthesisMetadata,
} from './types';

/**
 * Post one `SynthesisInteraction` feedback event. Returns `true` on
 * success, `false` if the gate refused or the network call failed —
 * callers can use the return for telemetry but should not surface it to
 * the user.
 */
export async function postSynthesisInteraction(
  synthesisId: string | undefined,
  recallId: string | undefined,
  interaction: SynthesisInteractionKind,
  metadata?: SynthesisMetadata,
): Promise<boolean> {
  // synthesis_id provenance gate — see §8 invariant 9. Skip silently.
  if (
    typeof synthesisId !== 'string' ||
    synthesisId.length === 0 ||
    typeof recallId !== 'string' ||
    recallId.length === 0
  ) {
    return false;
  }

  const body: FeedbackPayload = {
    kind: 'synthesis_interaction',
    synthesis_id: synthesisId,
    recall_id: recallId,
    interaction,
    ...(metadata ? { metadata } : {}),
  };

  try {
    // Result shape is `{ emitted: number }` per the server, but we don't
    // surface it — observability fire-and-forget pattern.
    await apiPost<{ emitted: number }>('/api/feedback', body);
    return true;
  } catch (err) {
    console.warn('[feedback] synthesis_interaction POST failed:', err);
    return false;
  }
}

/**
 * v0.27 — post one concept-summary interaction event (Cap A mirror of Cap B's
 * D-direction loop). Targets the dedicated `/api/feedback/concept-summary`
 * route the v0.27 backend exposes via inventory dispatch.
 *
 * Failure policy mirrors `postSynthesisInteraction`: silent, `console.warn`,
 * never bubbles to the user. Both correlation ids must be non-empty strings —
 * the gate refuses partial events so the consumer's stats stay honest.
 */
export async function postConceptSummaryFeedback(
  conceptId: string | undefined,
  recallId: string | undefined,
  interaction: ConceptSummaryInteractionKind,
  metadata?: ConceptSummaryMetadata,
  ids?: { conceptSummaryId?: string | null; livingSummaryId?: string | null },
): Promise<boolean> {
  if (
    typeof conceptId !== 'string' ||
    conceptId.length === 0 ||
    typeof recallId !== 'string' ||
    recallId.length === 0
  ) {
    return false;
  }

  const livingSummaryId =
    typeof ids?.livingSummaryId === 'string' && ids.livingSummaryId.length > 0
      ? ids.livingSummaryId
      : undefined;
  const conceptSummaryId =
    typeof ids?.conceptSummaryId === 'string' && ids.conceptSummaryId.length > 0
      ? ids.conceptSummaryId
      : livingSummaryId;

  const body: ConceptSummaryInteractionEvent = {
    concept_id: conceptId,
    recall_id: recallId,
    ...(conceptSummaryId ? { concept_summary_id: conceptSummaryId } : {}),
    ...(livingSummaryId ? { living_summary_id: livingSummaryId } : {}),
    interaction,
    ...(metadata ? { metadata } : {}),
  };

  try {
    await apiPost<{ ok: true } | { error: string }>(
      '/api/feedback/concept_summary',
      body,
    );
    return true;
  } catch (err) {
    console.warn('[feedback] concept_summary POST failed:', err);
    return false;
  }
}
