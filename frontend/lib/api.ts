// Typed client for the FastAPI backend (proxied at /api by next.config rewrites).

export type Hit = {
  video_id: string;
  frame_id: number;
  rank?: number;
  score?: number;
  rrf_score?: number;
  source?: string;
  sources?: string[];
  segment_id?: string;
  caption?: string;
  timestamp?: number;
  filename?: string;
  source_folder?: string;
  image_path?: string; // e.g. "/image?video_id=..&frame_id=.."
};

export type SearchResponse = { query: string; count: number; results: Hit[] };
export type AgentPlan = {
  semantic_query: string;
  module_queries: Record<string, string>;
  reasoning?: string;
};
export type AgentResponse = SearchResponse & { plan: AgentPlan };
export type VideoInfo = {
  video_id: string;
  fps: number;
  frame_count: number;
  duration: number;
  width: number;
  height: number;
};

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`/api${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(`${path} failed: ${res.status} ${await res.text()}`);
  return res.json();
}

export async function getModules(): Promise<string[]> {
  const res = await fetch("/api/modules");
  if (!res.ok) throw new Error("failed to load modules");
  return (await res.json()).modules;
}

export type Scope = { include?: string[]; exclude?: string[] };

export function searchSemantic(query: string, topK: number, scope: Scope = {}) {
  return post<SearchResponse>("/search", {
    query,
    top_k: topK,
    include_videos: scope.include ?? [],
    exclude_videos: scope.exclude ?? [],
  });
}

export function searchHybrid(
  query: string,
  modules: Record<string, string>,
  topK: number,
  scope: Scope = {}
) {
  return post<SearchResponse>("/search/hybrid", {
    query,
    modules,
    top_k: topK,
    include_videos: scope.include ?? [],
    exclude_videos: scope.exclude ?? [],
  });
}

export function agentSearch(query: string, topK: number, scope: Scope = {}) {
  return post<AgentResponse>("/agent/search", {
    query,
    top_k: topK,
    include_videos: scope.include ?? [],
    exclude_videos: scope.exclude ?? [],
  });
}

export function getVideoInfo(videoId: string) {
  return fetch(`/api/video/${videoId}/info`).then((r) => {
    if (!r.ok) throw new Error("video info failed");
    return r.json() as Promise<VideoInfo>;
  });
}

export type SubmissionResult = {
  csv: string | null;
  errors: string[];
  warnings: string[];
  row_count: number;
};

export function validateSubmission(
  queryType: string,
  rows: Record<string, unknown>[],
  eventCount = 1
) {
  return post<SubmissionResult>("/submission/validate", {
    query_type: queryType,
    rows,
    event_count: eventCount,
  });
}

export function prefetch(items: { video_id: string; frame_id: number }[]): void {
  // Fire-and-forget: warm the server thumbnail cache for the returned results.
  fetch("/api/prefetch", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ items }),
  }).catch(() => {});
}

export function imageSrc(hit: Hit): string {
  if (hit.image_path) return `/api${hit.image_path}`;
  return `/api/image?video_id=${encodeURIComponent(hit.video_id)}&frame_id=${hit.frame_id}`;
}

export function videoSrc(videoId: string): string {
  return `/api/video/${encodeURIComponent(videoId)}`;
}
