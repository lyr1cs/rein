import type { ProviderRole } from '../../api/types';

export const ROLE_ORDER: ProviderRole[] = ['embed', 'write', 'search', 'answer', 'judge', 'cloud'];

export interface RoleCopy {
  title: string;
  icon: string;
  what: string;
  runs: string;
  without: string;
  /** CLI commands to repoint this role; `{vendor}` / `{model}` / `{key}` / `{dims}` are placeholders. */
  cli: string[];
  toml: string;
}

export const ROLE_COPY: Record<ProviderRole, RoleCopy> = {
  embed: {
    title: 'Vectorize memories',
    icon: '🧭',
    what: 'Turns every memory into a vector so recall can match meaning, not just words.',
    runs: 'On every store and on every recall query.',
    without: 'The vector channel is off; recall is keyword-only.',
    cli: [
      'rein providers vendors',
      'rein providers set embedding --provider {vendor} --model {model} --dimensions {dims}',
      'rein keys set {key}',
      'rein warmup --reembed-all',
    ],
    toml: '[embedding]\nprovider = "{vendor}"\ndimensions = {dims}\n\n[embedding.openai]\nmodel = "{model}"',
  },
  write: {
    title: 'Write memories',
    icon: '✍️',
    what: 'Reads your sessions and tool output and writes them down as memories; decides whether a new fact merges into an old one.',
    runs: 'On every hook / proxy capture, in the async worker, on merge gray zones, and when a canonical hits its size cap.',
    without: 'Falls back to rule-based extraction; no LLM merge decisions; recompression stays off.',
    cli: ['rein providers set extract --provider {vendor} --model {model}', 'rein keys set {key}'],
    toml: '[extract]\nprovider = "{vendor}"\n\n[extract.openai]\nmodel = "{model}"',
  },
  search: {
    title: 'Sharpen search',
    icon: '🔍',
    what: 'Rewrites your question into variants and re-orders the top results.',
    runs: 'On every recall (expansion) and on the top-N when no strong signal decided already (rerank).',
    without: 'No expansion and no rerank; plain fusion only.',
    cli: [
      'rein providers set query_expansion --provider {vendor} --model {model}',
      'rein providers set reranker --provider {vendor}',
      'rein keys set {key}',
    ],
    toml: '[query_expansion]\nprovider = "{vendor}"\n\n[query_expansion.openai]\nmodel = "{model}"\n\n[search]\nllm_reranker = "{vendor}"',
  },
  answer: {
    title: 'Answer & summarize',
    icon: '🧠',
    what: 'Writes a narrative answer from recalled memories and keeps living summaries for concepts and cold archives.',
    runs: 'On recall with synthesize, and in the slow-channel refresh passes.',
    without: 'Recall returns raw memories only; summaries stop refreshing.',
    cli: [
      'rein providers set ars --provider {vendor}',
      'rein providers set llm --provider {vendor} --model {model}',
      'rein keys set {key}',
    ],
    toml: '[ars]\nllm_backend = "{vendor}"\n\n[llm]\nprovider = "{vendor}"\n\n[llm.openai]\nmodel = "{model}"',
  },
  judge: {
    title: 'Grade & learn',
    icon: '⚖️',
    what: 'Grades each answer against its sources so the adaptive loop learns which settings work.',
    runs: 'In the adaptive tick (every 15 minutes) and the nightly calibration.',
    without: 'No judge events; calibration stalls at zero pairs.',
    cli: ['rein providers set judge --provider {vendor} --model {model}', 'rein keys set {key}'],
    toml: '[ars.llm_judge]\nprovider = "{vendor}"\n\n[ars.llm_judge.openai]\nmodel = "{model}"',
  },
  cloud: {
    title: 'Cross-check in the cloud',
    icon: '☁️',
    what: 'Optional second opinion from Supermemory to cross-validate recall.',
    runs: 'On recall, when enabled.',
    without: 'Cross-validation is partial.',
    cli: ['rein keys set SUPERMEMORY_CC_API_KEY'],
    toml: '[sync]\nsupermemory_enabled = true',
  },
};

export interface VendorCopy {
  label: string;
  keyUrl?: string;
}

/** Keyed by the provider name the report uses. Unknown names fall back to "OpenAI-compatible endpoint". */
export const VENDOR_COPY: Record<string, VendorCopy> = {
  google: { label: 'Google AI Studio key', keyUrl: 'https://aistudio.google.com/app/apikey' },
  openai: { label: 'OpenAI platform key', keyUrl: 'https://platform.openai.com/api-keys' },
  perplexity: { label: 'Perplexity API key', keyUrl: 'https://www.perplexity.ai/settings/api' },
  deepseek: { label: 'DeepSeek platform key', keyUrl: 'https://platform.deepseek.com/api_keys' },
  openrouter: { label: 'OpenRouter key', keyUrl: 'https://openrouter.ai/keys' },
  groq: { label: 'Groq key', keyUrl: 'https://console.groq.com/keys' },
  mistral: { label: 'Mistral key', keyUrl: 'https://console.mistral.ai/api-keys' },
  xai: { label: 'xAI key', keyUrl: 'https://console.x.ai' },
  together: { label: 'Together AI key', keyUrl: 'https://api.together.ai/settings/api-keys' },
  fireworks: { label: 'Fireworks key', keyUrl: 'https://fireworks.ai/account/api-keys' },
  voyage: { label: 'Voyage AI key', keyUrl: 'https://dashboard.voyageai.com/api-keys' },
  jina: { label: 'Jina AI key', keyUrl: 'https://jina.ai/api-dashboard/' },
  cohere: { label: 'Cohere key', keyUrl: 'https://dashboard.cohere.com/api-keys' },
  anthropic: { label: 'Anthropic key (OpenAI-compatible endpoint)', keyUrl: 'https://console.anthropic.com/settings/keys' },
  azure: { label: 'Azure OpenAI key' },
  nvidia: { label: 'NVIDIA NIM key', keyUrl: 'https://build.nvidia.com' },
  cerebras: { label: 'Cerebras key', keyUrl: 'https://cloud.cerebras.ai' },
  huggingface: { label: 'Hugging Face token', keyUrl: 'https://huggingface.co/settings/tokens' },
  dashscope: { label: '阿里云百炼 DashScope key', keyUrl: 'https://bailian.console.aliyun.com' },
  zhipu: { label: '智谱 BigModel key', keyUrl: 'https://open.bigmodel.cn' },
  siliconflow: { label: '硅基流动 key', keyUrl: 'https://cloud.siliconflow.cn' },
  moonshot: { label: 'Moonshot / Kimi key', keyUrl: 'https://platform.moonshot.cn' },
  volcengine: { label: '火山方舟 key', keyUrl: 'https://console.volcengine.com/ark' },
  supermemory: { label: 'Supermemory key', keyUrl: 'https://supermemory.ai' },
  ollama: { label: 'Ollama (local, no key)' },
  lmstudio: { label: 'LM Studio (local, no key)' },
  vllm: { label: 'vLLM (local, no key)' },
  omlx: { label: 'Local OpenAI-compatible server (no key)' },
};

export const STATUS_LABEL: Record<string, string> = {
  ready: 'ready',
  missing: 'missing key',
  missing_credential: 'missing key',
  disabled: 'off',
};
