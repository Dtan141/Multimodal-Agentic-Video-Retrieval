"use client";

import { useEffect, useState } from "react";
import {
  agentSearch,
  getModules,
  prefetch,
  searchHybrid,
  type AgentPlan,
  type Hit,
} from "@/lib/api";
import ResultGrid from "@/components/ResultGrid";
import VideoInspector from "@/components/VideoInspector";
import CsvBuilder, { type CsvRow } from "@/components/CsvBuilder";

type Mode = "agent" | "manual";

export default function Home() {
  const [mode, setMode] = useState<Mode>("agent");
  const [query, setQuery] = useState("");
  const [topK, setTopK] = useState(30);
  const [modules, setModules] = useState<string[]>([]);
  const [manualKw, setManualKw] = useState<Record<string, string>>({});

  const [results, setResults] = useState<Hit[]>([]);
  const [plan, setPlan] = useState<AgentPlan | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const [inspector, setInspector] = useState<{ videoId: string; frame: number } | null>(null);
  const [openName, setOpenName] = useState("");

  const openByName = () => {
    const name = openName.trim().replace(/\.mp4$/i, "");
    if (name) setInspector({ videoId: name, frame: 0 });
  };

  // CSV state (lifted so inspector picks can append)
  const [queryType, setQueryType] = useState("KIS");
  const [eventCount, setEventCount] = useState(4);
  const [rows, setRows] = useState<CsvRow[]>([]);
  const [csvFilename, setCsvFilename] = useState("");

  const uploadQueryTxt = async (file: File) => {
    const text = await file.text();
    setQuery(text.trim());
    // "query-1-kis.txt" -> csv filename "query-1-kis"
    setCsvFilename(file.name.replace(/\.txt$/i, ""));
  };

  // video scope filter (persists across searches)
  const [includeVideos, setIncludeVideos] = useState<string[]>([]);
  const [excludeVideos, setExcludeVideos] = useState<string[]>([]);
  const [incInput, setIncInput] = useState("");
  const [excInput, setExcInput] = useState("");
  const [selectedKeys, setSelectedKeys] = useState<Set<string>>(new Set());

  const addSpec = (list: string[], setList: (v: string[]) => void, spec: string) => {
    const s = spec.trim().replace(/\.mp4$/i, "");
    if (s && !list.includes(s)) setList([...list, s]);
  };
  const toggleSel = (key: string) => {
    const next = new Set(selectedKeys);
    if (next.has(key)) next.delete(key);
    else next.add(key);
    setSelectedKeys(next);
  };
  const addSelectedTo = (which: "inc" | "exc") => {
    const vids = Array.from(new Set(Array.from(selectedKeys).map((k) => k.split("#")[0])));
    if (which === "inc") setIncludeVideos(Array.from(new Set([...includeVideos, ...vids])));
    else setExcludeVideos(Array.from(new Set([...excludeVideos, ...vids])));
    setSelectedKeys(new Set());
  };

  useEffect(() => {
    getModules().then(setModules).catch(() => setModules(["caption", "ocr", "object", "asr"]));
  }, []);

  const runSearch = async () => {
    const hasModuleKw = Object.values(manualKw).some((v) => v?.trim());
    const canSearch = mode === "agent" ? !!query.trim() : !!query.trim() || hasModuleKw;
    if (!canSearch) return;
    setLoading(true);
    setError(null);
    setPlan(null);
    setSelectedKeys(new Set());
    const scope = { include: includeVideos, exclude: excludeVideos };
    try {
      let hits: Hit[];
      if (mode === "agent") {
        const res = await agentSearch(query, topK, scope);
        hits = res.results;
        setPlan(res.plan);
      } else {
        const mods: Record<string, string> = {};
        for (const m of modules) if (manualKw[m]?.trim()) mods[m] = manualKw[m].trim();
        const res = await searchHybrid(query, mods, topK, scope);
        hits = res.results;
      }
      setResults(hits);
      prefetch(hits.map((h) => ({ video_id: h.video_id, frame_id: h.frame_id })));
    } catch (e) {
      setError(String(e));
      setResults([]);
    } finally {
      setLoading(false);
    }
  };

  const pickFrame = (videoId: string, frame: number) => {
    if (queryType === "TRAKE") {
      const next = [...rows];
      const slots = Array.from({ length: eventCount }, (_, i) => `frame_${i + 1}`);
      const last = next[next.length - 1];
      const empty = last && slots.find((s) => !last[s]);
      if (last && empty) last[empty] = frame;
      else next.push({ video_name: videoId, frame_1: frame });
      setRows(next);
    } else {
      setRows([...rows, { video_name: videoId, frame_id: frame }]);
    }
  };

  return (
    <main className="mx-auto max-w-[1700px] px-4 py-5">
      <header className="mb-4">
        <h1 className="text-xl font-bold">AIC 2026 — Truy xuất Multimedia</h1>
        <p className="text-xs text-neutral-500">
          Semantic (SigLIP2) + metadata (OCR/ASR/Object/Caption) hybrid search · agent hỗ trợ
        </p>
      </header>

      <div className="flex flex-col gap-6 lg:flex-row">
        {/* ================= LEFT: query + submission ================= */}
        <aside className="space-y-4 lg:sticky lg:top-4 lg:max-h-[calc(100vh-1.5rem)] lg:w-[380px] lg:shrink-0 lg:self-start lg:overflow-y-auto lg:pr-1">
          {/* Mode toggle */}
          <div className="inline-flex rounded-lg border border-neutral-300 p-1 dark:border-neutral-700">
            {(["agent", "manual"] as Mode[]).map((m) => (
              <button
                key={m}
                onClick={() => setMode(m)}
                className={`rounded-md px-4 py-1.5 text-sm font-medium ${
                  mode === m ? "bg-brand text-brand-fg" : "text-neutral-500"
                }`}
              >
                {m === "agent" ? "🤖 Agent" : "🎚️ Thủ công"}
              </button>
            ))}
          </div>

          {/* Search box */}
          <div className="space-y-2">
            <textarea
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  runSearch();
                }
              }}
              rows={2}
              placeholder={
                mode === "agent"
                  ? "Nhập câu truy vấn (tiếng Việt ok)…"
                  : "Mô tả cảnh (semantic) — có thể để trống nếu dùng module bên dưới…"
              }
              className="w-full resize-y rounded-lg border border-neutral-300 bg-transparent px-3 py-2 text-sm dark:border-neutral-700"
            />
            <div className="flex items-center gap-2">
              <label className="flex items-center gap-1 text-xs text-neutral-500">
                top-k
                <input
                  type="number"
                  min={1}
                  max={200}
                  value={topK}
                  onChange={(e) => setTopK(Number(e.target.value))}
                  className="w-16 rounded-lg border border-neutral-300 bg-transparent px-2 py-1.5 dark:border-neutral-700"
                />
              </label>
              <label
                className="flex cursor-pointer items-center rounded-lg border border-neutral-300 px-2 py-1.5 text-sm dark:border-neutral-700"
                title="Tải file .txt chứa query (tên file → tên CSV)"
              >
                📄 .txt
                <input
                  type="file"
                  accept=".txt,text/plain"
                  className="hidden"
                  onChange={(e) => {
                    const f = e.target.files?.[0];
                    if (f) uploadQueryTxt(f);
                    e.target.value = "";
                  }}
                />
              </label>
              <button
                onClick={runSearch}
                disabled={loading}
                className="ml-auto rounded-lg bg-brand px-5 py-2 text-sm font-semibold text-brand-fg disabled:opacity-50"
              >
                {loading ? "Đang tìm…" : "Tìm kiếm"}
              </button>
            </div>
            {csvFilename && (
              <p className="text-xs text-neutral-500">
                Tên file CSV: <b>{csvFilename}.csv</b>
              </p>
            )}
          </div>

          {/* Manual module controls */}
          {mode === "manual" && (
            <div className="grid grid-cols-2 gap-2">
              {modules.map((m) => (
                <label key={m} className="text-sm">
                  <span className="text-neutral-500">{m}</span>
                  <input
                    value={manualKw[m] ?? ""}
                    onChange={(e) => setManualKw({ ...manualKw, [m]: e.target.value })}
                    placeholder={`từ khóa ${m}`}
                    className="mt-1 w-full rounded border border-neutral-300 bg-transparent px-2 py-1 text-sm dark:border-neutral-700"
                  />
                </label>
              ))}
            </div>
          )}

          {/* Video scope filter */}
          <div className="grid gap-3 rounded-lg border border-neutral-200 bg-white p-3 dark:border-neutral-800 dark:bg-neutral-900">
            {(
              [
                ["✅ Ưu tiên (include)", includeVideos, setIncludeVideos, incInput, setIncInput, "green"],
                ["🚫 Loại trừ (exclude)", excludeVideos, setExcludeVideos, excInput, setExcInput, "red"],
              ] as const
            ).map(([label, list, setList, input, setInput, color]) => (
              <div key={label}>
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-sm font-medium">{label}</span>
                  {list.length > 0 && (
                    <button onClick={() => setList([])} className="text-xs text-neutral-400 hover:text-red-500">
                      xóa hết
                    </button>
                  )}
                </div>
                <div className="flex gap-1">
                  <input
                    value={input}
                    onChange={(e) => setInput(e.target.value)}
                    onKeyDown={(e) => {
                      if (e.key === "Enter") {
                        addSpec(list, setList, input);
                        setInput("");
                      }
                    }}
                    placeholder="L21 hoặc L21_V018"
                    className="flex-1 rounded border border-neutral-300 bg-transparent px-2 py-1 text-sm dark:border-neutral-700"
                  />
                  <button
                    onClick={() => {
                      addSpec(list, setList, input);
                      setInput("");
                    }}
                    className="rounded bg-neutral-200 px-2 py-1 text-sm dark:bg-neutral-800"
                  >
                    +
                  </button>
                </div>
                <div className="mt-2 flex flex-wrap gap-1">
                  {list.map((v) => (
                    <span
                      key={v}
                      className={`flex items-center gap-1 rounded px-2 py-0.5 text-xs ${
                        color === "green"
                          ? "bg-green-100 text-green-800 dark:bg-green-900/40 dark:text-green-300"
                          : "bg-red-100 text-red-800 dark:bg-red-900/40 dark:text-red-300"
                      }`}
                    >
                      {v}
                      <button onClick={() => setList(list.filter((x) => x !== v))} className="hover:opacity-70">
                        ✕
                      </button>
                    </span>
                  ))}
                </div>
              </div>
            ))}
          </div>

          {/* Open a specific video by name */}
          <div className="flex flex-wrap items-center gap-2 rounded-lg border border-neutral-200 bg-white p-2 dark:border-neutral-800 dark:bg-neutral-900">
            <span className="text-sm text-neutral-500">🎬 Mở video:</span>
            <input
              value={openName}
              onChange={(e) => setOpenName(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && openByName()}
              placeholder="L21_V018"
              className="w-32 rounded border border-neutral-300 bg-transparent px-2 py-1 text-sm dark:border-neutral-700"
            />
            <button onClick={openByName} className="rounded bg-neutral-800 px-3 py-1 text-sm text-white dark:bg-neutral-200 dark:text-neutral-900">
              Mở
            </button>
          </div>

          {/* CSV builder */}
          <CsvBuilder
            queryType={queryType}
            setQueryType={setQueryType}
            eventCount={eventCount}
            setEventCount={setEventCount}
            rows={rows}
            setRows={setRows}
            filename={csvFilename}
            setFilename={setCsvFilename}
          />
        </aside>

        {/* ================= RIGHT: results ================= */}
        <section className="min-w-0 flex-1">
          {plan && (
            <div className="mb-4 rounded-lg border border-brand/30 bg-brand/5 p-3 text-sm">
              <p>
                <b>Semantic:</b> {plan.semantic_query}
              </p>
              <p className="mt-1">
                <b>Tools:</b>{" "}
                {Object.entries(plan.module_queries).map(([k, v]) => (
                  <span key={k} className="mr-2 rounded bg-brand/10 px-1.5 py-0.5 text-brand">
                    {k}: {v}
                  </span>
                ))}
              </p>
              {plan.reasoning && <p className="mt-1 text-neutral-500">{plan.reasoning}</p>}
            </div>
          )}

          {error && <p className="mb-3 text-red-500">{error}</p>}

          {results.length > 0 ? (
            <div className="mb-2 flex flex-wrap items-center gap-3 text-sm text-neutral-500">
              <span>{results.length} kết quả — bấm ảnh để mở video, tick để chọn</span>
              {selectedKeys.size > 0 && (
                <span className="flex items-center gap-2">
                  <b>{selectedKeys.size} đã chọn</b>
                  <button onClick={() => addSelectedTo("inc")} className="rounded bg-green-600 px-2 py-0.5 text-white">
                    + Ưu tiên
                  </button>
                  <button onClick={() => addSelectedTo("exc")} className="rounded bg-red-600 px-2 py-0.5 text-white">
                    + Loại trừ
                  </button>
                  <button onClick={() => setSelectedKeys(new Set())} className="rounded bg-neutral-300 px-2 py-0.5 dark:bg-neutral-700">
                    Bỏ chọn
                  </button>
                </span>
              )}
            </div>
          ) : (
            !loading && <p className="text-sm text-neutral-400">Nhập truy vấn bên trái rồi bấm Tìm kiếm.</p>
          )}

          <ResultGrid
            results={results}
            onOpen={(h) => setInspector({ videoId: h.video_id, frame: h.frame_id })}
            selectedKeys={selectedKeys}
            onToggle={toggleSel}
          />
        </section>
      </div>

      {inspector && (
        <VideoInspector
          videoId={inspector.videoId}
          initialFrame={inspector.frame}
          onClose={() => setInspector(null)}
          onPickFrame={pickFrame}
          csvRows={rows}
          queryType={queryType}
          onRemoveRow={(i) => setRows(rows.filter((_, idx) => idx !== i))}
        />
      )}
    </main>
  );
}
