/**
 * Read the user's polling cadence (seconds) from localStorage and return the
 * value in milliseconds. Centralized so Brain.tsx, Graph.tsx, and the
 * `useApi.ts` react-query hooks all read the same setting (changeable from
 * the Settings page) without each page rolling its own copy.
 *
 * Falls back to 5000ms when the value is missing, NaN, or non-positive.
 */
export function getPollingIntervalMs(): number {
  const saved = localStorage.getItem('rein_polling_interval');
  const seconds = saved ? parseInt(saved, 10) : 5;
  return Number.isFinite(seconds) && seconds > 0 ? seconds * 1000 : 5000;
}
