import { Suspense, lazy } from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import Layout from './components/Layout';

const Dashboard = lazy(() => import('./pages/Dashboard'));
const Brain = lazy(() => import('./pages/Brain'));
const Memories = lazy(() => import('./pages/Memories'));
const Adaptive = lazy(() => import('./pages/Adaptive'));
const Graph = lazy(() => import('./pages/Graph'));
const Timeline = lazy(() => import('./pages/Timeline'));
const Artifacts = lazy(() => import('./pages/Artifacts'));
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
              <Route path="/settings" element={<Settings />} />
            </Route>
          </Routes>
        </Suspense>
      </BrowserRouter>
    </QueryClientProvider>
  );
}
