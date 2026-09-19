"use client";

import { useState } from "react";
import { validateSubmission, type SubmissionResult } from "@/lib/api";

export type CsvRow = Record<string, string | number>;

type Props = {
  queryType: string;
  setQueryType: (t: string) => void;
  eventCount: number;
  setEventCount: (n: number) => void;
  rows: CsvRow[];
  setRows: (r: CsvRow[]) => void;
  filename: string;
  setFilename: (s: string) => void;
};

const TYPES = ["KIS", "Q&A", "TRAKE"];

export default function CsvBuilder({
  queryType,
  setQueryType,
  eventCount,
  setEventCount,
  rows,
  setRows,
  filename,
  setFilename,
}: Props) {
  const [result, setResult] = useState<SubmissionResult | null>(null);
  const [sharedAnswer, setSharedAnswer] = useState("");

  // range generation
  const [genVideo, setGenVideo] = useState("");
  const [genStart, setGenStart] = useState(0);
  const [genEnd, setGenEnd] = useState(0);
  const [genStep, setGenStep] = useState(30);
  const [genMsg, setGenMsg] = useState<string | null>(null);

  const columns =
    queryType === "KIS"
      ? ["video_name", "frame_id"]
      : queryType === "Q&A"
        ? ["video_name", "frame_id", "answer"]
        : ["video_name", ...Array.from({ length: eventCount }, (_, i) => `frame_${i + 1}`)];

  const update = (i: number, key: string, val: string) => {
    setRows(rows.map((r, idx) => (idx === i ? { ...r, [key]: val } : r)));
  };
  const addRow = () => setRows([...rows, { video_name: "" }]);
  const removeRow = (i: number) => setRows(rows.filter((_, idx) => idx !== i));
  const clearRows = () => {
    setRows([]);
    setResult(null);
  };
  const applyAnswerToAll = () => setRows(rows.map((r) => ({ ...r, answer: sharedAnswer })));

  const generate = () => {
    setGenMsg(null);
    const video = genVideo.trim().replace(/\.mp4$/i, "");
    if (!video) return setGenMsg("Nhập tên video trước.");
    if (genStep <= 0) return setGenMsg("Khoảng cách phải > 0.");
    if (genEnd < genStart) return setGenMsg("Frame end phải ≥ frame start.");

    const frames: number[] = [];
    for (let f = genStart; f <= genEnd; f += genStep) frames.push(f);
    if (frames.length === 0) return setGenMsg("Không tạo được frame nào.");

    if (queryType === "TRAKE") {
      if (frames.length !== eventCount)
        return setGenMsg(`TRAKE cần đúng ${eventCount} frame, dải tạo ra ${frames.length}.`);
      const row: CsvRow = { video_name: video };
      frames.forEach((f, i) => (row[`frame_${i + 1}`] = f));
      setRows([...rows, row]);
      setGenMsg(`Đã thêm 1 dòng TRAKE (${frames.length} frame).`);
      return;
    }

    const capped = frames.slice(0, 100);
    const newRows: CsvRow[] = capped.map((f): CsvRow =>
      queryType === "Q&A"
        ? { video_name: video, frame_id: f, answer: sharedAnswer }
        : { video_name: video, frame_id: f }
    );
    setRows(newRows);
    setGenMsg(`Đã sinh ${newRows.length} dòng (top-1 → top-${newRows.length})${frames.length > 100 ? ", cắt còn 100" : ""}.`);
  };

  const validate = async () => setResult(await validateSubmission(queryType, rows, eventCount));

  const download = () => {
    if (!result?.csv) return;
    let name = filename.trim() || `submission-${queryType.toLowerCase().replace("&", "")}`;
    if (!name.toLowerCase().endsWith(".csv")) name += ".csv";
    const blob = new Blob([result.csv], { type: "text/csv;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = name;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <section id="csv-builder" className="rounded-xl border border-neutral-200 bg-white p-4 dark:border-neutral-800 dark:bg-neutral-900">
      <h2 className="mb-3 text-lg font-semibold">Tạo file nộp (CSV)</h2>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <select
          value={queryType}
          onChange={(e) => {
            setQueryType(e.target.value);
            setRows([]);
            setResult(null);
          }}
          className="rounded border border-neutral-300 bg-transparent px-2 py-1 text-sm dark:border-neutral-700"
        >
          {TYPES.map((t) => (
            <option key={t} value={t}>
              {t}
            </option>
          ))}
        </select>
        {queryType === "TRAKE" && (
          <label className="flex items-center gap-1 text-sm">
            Số events
            <input
              type="number"
              min={1}
              max={20}
              value={eventCount}
              onChange={(e) => setEventCount(Math.max(1, Number(e.target.value)))}
              className="w-16 rounded border border-neutral-300 bg-transparent px-2 py-1 dark:border-neutral-700"
            />
          </label>
        )}
        <button onClick={addRow} className="rounded bg-neutral-200 px-3 py-1 text-sm dark:bg-neutral-800">
          + Dòng
        </button>
        <button onClick={clearRows} className="rounded bg-neutral-200 px-3 py-1 text-sm dark:bg-neutral-800">
          Xóa hết
        </button>
        <span className="text-xs text-neutral-500">{rows.length}/100 dòng</span>
      </div>

      {/* Range generation */}
      <div className="mb-3 flex flex-wrap items-end gap-2 rounded-lg bg-neutral-100 p-2 dark:bg-neutral-800">
        <span className="w-full text-xs font-medium text-neutral-500">
          Sinh nhanh theo dải (start → end, bước) — tối đa 100 dòng
        </span>
        <input
          value={genVideo}
          onChange={(e) => setGenVideo(e.target.value)}
          placeholder="video (L21_V018)"
          className="w-36 rounded border border-neutral-300 bg-transparent px-2 py-1 text-sm dark:border-neutral-700"
        />
        <label className="text-xs">
          start
          <input type="number" value={genStart} onChange={(e) => setGenStart(Number(e.target.value))} className="ml-1 w-20 rounded border border-neutral-300 bg-transparent px-2 py-1 dark:border-neutral-700" />
        </label>
        <label className="text-xs">
          end
          <input type="number" value={genEnd} onChange={(e) => setGenEnd(Number(e.target.value))} className="ml-1 w-20 rounded border border-neutral-300 bg-transparent px-2 py-1 dark:border-neutral-700" />
        </label>
        <label className="text-xs">
          bước
          <input type="number" min={1} value={genStep} onChange={(e) => setGenStep(Number(e.target.value))} className="ml-1 w-16 rounded border border-neutral-300 bg-transparent px-2 py-1 dark:border-neutral-700" />
        </label>
        <button onClick={generate} className="rounded bg-brand px-3 py-1 text-sm font-semibold text-brand-fg">
          Sinh dải
        </button>
        {genMsg && <span className="text-xs text-neutral-500">{genMsg}</span>}
      </div>

      {queryType === "Q&A" && (
        <div className="mb-3 flex flex-wrap items-center gap-2 rounded-lg bg-neutral-100 p-2 dark:bg-neutral-800">
          <span className="text-sm text-neutral-500">Answer chung:</span>
          <input
            value={sharedAnswer}
            onChange={(e) => setSharedAnswer(e.target.value)}
            maxLength={100}
            placeholder="Nhập 1 lần cho tất cả frame…"
            className="min-w-[240px] flex-1 rounded border border-neutral-300 bg-transparent px-2 py-1 text-sm dark:border-neutral-700"
          />
          <button onClick={applyAnswerToAll} className="rounded bg-brand px-3 py-1 text-sm font-semibold text-brand-fg">
            Điền cho tất cả ({rows.length})
          </button>
        </div>
      )}

      <div className="max-h-72 overflow-auto">
        <table className="w-full text-sm">
          <thead className="sticky top-0 bg-white dark:bg-neutral-900">
            <tr className="text-left text-neutral-500">
              {columns.map((c) => (
                <th key={c} className="px-2 py-1 font-medium">
                  {c}
                </th>
              ))}
              <th />
            </tr>
          </thead>
          <tbody>
            {rows.map((row, i) => (
              <tr key={i} className="border-t border-neutral-100 dark:border-neutral-800">
                {columns.map((c) => (
                  <td key={c} className="px-1 py-1">
                    <input
                      value={(row[c] ?? "") as string}
                      onChange={(e) => update(i, c, e.target.value)}
                      className="w-full rounded border border-neutral-200 bg-transparent px-2 py-1 dark:border-neutral-700"
                    />
                  </td>
                ))}
                <td>
                  <button onClick={() => removeRow(i)} className="px-2 text-neutral-400 hover:text-red-500">
                    ✕
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="mt-3 flex flex-wrap items-center gap-2">
        <input
          value={filename}
          onChange={(e) => setFilename(e.target.value)}
          placeholder="tên file (vd: query-1-kis)"
          className="w-56 rounded border border-neutral-300 bg-transparent px-2 py-1.5 text-sm dark:border-neutral-700"
        />
        <button onClick={validate} className="rounded bg-brand px-4 py-1.5 text-sm font-semibold text-brand-fg">
          Kiểm tra
        </button>
        {result?.csv && (
          <button onClick={download} className="rounded bg-green-600 px-4 py-1.5 text-sm font-semibold text-white">
            Tải CSV
          </button>
        )}
      </div>

      {result && (
        <div className="mt-3 text-sm">
          {result.errors.map((e, i) => (
            <p key={i} className="text-red-500">
              • {e}
            </p>
          ))}
          {result.warnings.map((w, i) => (
            <p key={i} className="text-amber-500">
              ⚠ {w}
            </p>
          ))}
          {result.csv && (
            <pre className="mt-2 overflow-x-auto rounded bg-neutral-100 p-2 font-mono text-xs dark:bg-neutral-800">
              {result.csv}
            </pre>
          )}
        </div>
      )}
    </section>
  );
}
