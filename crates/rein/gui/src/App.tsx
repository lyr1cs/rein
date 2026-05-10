import { Suspense, lazy, Component } from 'react';
import type { ReactNode, ErrorInfo } from 'react';
import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout';

/**
 * Top-level ErrorBoundary — catches render-time exceptions from any lazy
 * page so a single-page crash doesn't blank the whole app.
 *
 * Class-component intentional: React 19 still has no hook equivalent for
 * `componentDidCatch` / `getDerivedStateFromError` (the official ESLint
 * rule allows this exact pattern). When a render throws,
 * `getDerivedStateFromError` updates state to the captured error, which
 * `render` reads to swap in the fallback UI.
 *
 * "Go Home" performs a full document reload (`window.location.href = '/'`)
 * rather than `navigate('/')` so the QueryClient + lazy-loaded chunks
 * re-initialize from a clean slate — recoverable from caches that
 * captured the bad data are the most common cause of repeat-crashes.
 *
 * L7 (v0.26 cleanup): expanded the previous one-liner into a proper
 * docstring covering both the "why class component" and the "why full
 * reload on Go Home" decisions, since both have come up in subsequent
 * GUI audits.
 */
class ErrorBoundary extends Component<{ children: ReactNode }, { error: Error | null }> {
  state = { error: null as Error | null };
  static getDerivedStateFromError(error: Error) { return { error }; }
  componentDidCatch(error: Error, info: ErrorInfo) { console.error('ErrorBoundary caught:', error, info); }
  render() {
    if (this.state.error) {
      return (
        <div className="flex h-screen flex-col items-center justify-center gap-4 bg-[var(--bg-primary)] text-[var(--text-primary)]">
          <h1 className="text-xl font-bold text-[var(--error)]">Something went wrong</h1>
          <pre className="max-w-lg overflow-auto rounded bg-[var(--bg-secondary)] p-4 text-sm text-[var(--text-muted)]">
            {this.state.error.message}
          </pre>
          <button className="rounded bg-[var(--accent)] px-4 py-2 text-white" onClick={() => { this.setState({ error: null }); window.location.href = '/'; }}>
            Go Home
          </button>
        </div>
      );
    }
    return this.props.children;
  }
}

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Brain = lazy(() => import('./pages/Brain'));
const Memories = lazy(() => import('./pages/Memories'));
const Adaptive = lazy(() => import('./pages/Adaptive'));
const Graph = lazy(() => import('./pages/Graph'));
const Timeline = lazy(() => import('./pages/Timeline'));
const Artifacts = lazy(() => import('./pages/Artifacts'));
const Provenance = lazy(() => import('./pages/Provenance'));
const SynthesisLab = lazy(() => import('./pages/SynthesisLab'));
const Connectors = lazy(() => import('./pages/Connectors'));
const Settings = lazy(() => import('./pages/Settings'));

const queryClient = new QueryClient({
  defaultOptions: {
    queries: {
      staleTime: 3000,
      retry: 1,
    },
  },
});

export default function App() {
  return (
    <ErrorBoundary>
      <QueryClientProvider client={queryClient}>
        <BrowserRouter>
          <Suspense
            fallback={
              <div className="flex h-screen items-center justify-center bg-[var(--bg-primary)] text-[var(--text-muted)]">
                Loading interface...
              </div>
            }
          >
            <Routes>
              <Route element={<Layout />}>
                <Route path="/" element={<Dashboard />} />
                <Route path="/brain" element={<Brain />} />
                <Route path="/memories" element={<Memories />} />
                <Route path="/adaptive" element={<Adaptive />} />
                <Route path="/graph" element={<Graph />} />
                <Route path="/timeline" element={<Timeline />} />
                <Route path="/artifacts" element={<Artifacts />} />
                <Route path="/provenance" element={<Provenance />} />
                <Route path="/synthesis-lab" element={<SynthesisLab />} />
                <Route path="/connectors" element={<Connectors />} />
                <Route path="/settings" element={<Settings />} />
                <Route path="*" element={<Navigate to="/" replace />} />
              </Route>
            </Routes>
          </Suspense>
        </BrowserRouter>
      </QueryClientProvider>
    </ErrorBoundary>
  );
}
