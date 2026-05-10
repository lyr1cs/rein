import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { apiGet, apiPost } from '../api/client';

type OAuthClient = {
  client_id: string;
  client_name: string;
  registered_at: number;
  last_used_at: number | null;
  revoked_at: number | null;
  active_grants: number;
};

type ClientsResponse = {
  clients: OAuthClient[];
};

function formatTime(value: number | null) {
  if (!value) return 'never';
  return new Date(value * 1000).toLocaleString();
}

export default function Connectors() {
  const queryClient = useQueryClient();
  const { data, isLoading, error } = useQuery({
    queryKey: ['oauth-clients'],
    queryFn: () => apiGet<ClientsResponse>('/api/oauth/clients'),
  });
  const revoke = useMutation({
    mutationFn: (clientId: string) =>
      apiPost<{ revoked: boolean }>(`/api/oauth/clients/${encodeURIComponent(clientId)}/revoke`, {}),
    onSuccess: () => queryClient.invalidateQueries({ queryKey: ['oauth-clients'] }),
  });

  return (
    <div className="p-6 space-y-5">
      <div>
        <h1 className="text-xl font-semibold text-[var(--text-primary)]">Connectors</h1>
      </div>

      {isLoading && <div className="text-sm text-[var(--text-muted)]">Loading connectors...</div>}
      {error && <div className="text-sm text-[var(--hot)]">{(error as Error).message}</div>}

      <div className="overflow-hidden border border-[var(--border)] bg-[var(--bg-primary)]">
        <table className="w-full text-left text-sm">
          <thead className="border-b border-[var(--border)] bg-[var(--bg-secondary)] text-xs uppercase text-[var(--text-muted)]">
            <tr>
              <th className="px-4 py-3 font-medium">Name</th>
              <th className="px-4 py-3 font-medium">Client ID</th>
              <th className="px-4 py-3 font-medium">Registered</th>
              <th className="px-4 py-3 font-medium">Last Used</th>
              <th className="px-4 py-3 font-medium">Active Grants</th>
              <th className="px-4 py-3 font-medium">Status</th>
              <th className="px-4 py-3 font-medium" />
            </tr>
          </thead>
          <tbody>
            {(data?.clients ?? []).map((client) => (
              <tr key={client.client_id} className="border-b border-[var(--border)] last:border-b-0">
                <td className="px-4 py-3 text-[var(--text-primary)]">{client.client_name}</td>
                <td className="px-4 py-3 font-mono text-xs text-[var(--text-muted)]">{client.client_id.slice(0, 12)}...</td>
                <td className="px-4 py-3 text-[var(--text-secondary)]">{formatTime(client.registered_at)}</td>
                <td className="px-4 py-3 text-[var(--text-secondary)]">{formatTime(client.last_used_at)}</td>
                <td className="px-4 py-3 font-mono text-[var(--text-secondary)]">{client.active_grants}</td>
                <td className="px-4 py-3 text-[var(--text-secondary)]">{client.revoked_at ? 'revoked' : 'active'}</td>
                <td className="px-4 py-3 text-right">
                  <button
                    type="button"
                    disabled={Boolean(client.revoked_at) || revoke.isPending}
                    onClick={() => revoke.mutate(client.client_id)}
                    className="border border-[var(--border)] px-3 py-1.5 text-xs text-[var(--text-primary)] hover:bg-[var(--bg-secondary)] disabled:cursor-not-allowed disabled:opacity-40"
                  >
                    Revoke
                  </button>
                </td>
              </tr>
            ))}
            {!isLoading && (data?.clients ?? []).length === 0 && (
              <tr>
                <td className="px-4 py-8 text-center text-[var(--text-muted)]" colSpan={7}>
                  No connectors registered.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
