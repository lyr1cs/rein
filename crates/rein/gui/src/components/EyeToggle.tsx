interface Props {
  shown: boolean;
  onToggle: () => void;
  label?: string;
  className?: string;
}

/** Small eye / eye-off button used for secrets (token field, key cards). */
export default function EyeToggle({ shown, onToggle, label = 'secret', className = '' }: Props) {
  return (
    <button
      type="button"
      onClick={onToggle}
      aria-label={shown ? `Hide ${label}` : `Show ${label}`}
      aria-pressed={shown}
      className={`inline-flex items-center justify-center rounded p-1 text-[var(--text-muted)] transition-colors hover:text-[var(--text-primary)] ${className}`}
    >
      {shown ? (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M3 3l18 18" />
          <path d="M10.6 10.6a2 2 0 0 0 2.8 2.8" />
          <path d="M9.9 5.1A10.9 10.9 0 0 1 12 5c5 0 9 4 10 7-.4 1.2-1.2 2.5-2.3 3.6" />
          <path d="M6.6 6.6C4.4 8 2.8 10 2 12c1 3 5 7 10 7 1.5 0 2.9-.3 4.1-.9" />
        </svg>
      ) : (
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M2 12s4-7 10-7 10 7 10 7-4 7-10 7S2 12 2 12z" />
          <circle cx="12" cy="12" r="3" />
        </svg>
      )}
    </button>
  );
}
