"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { getVideoInfo, videoSrc, type VideoInfo } from "@/lib/api";
import type { CsvRow } from "@/components/CsvBuilder";

type Props = {
  videoId: string;
  initialFrame?: number;
  onClose: () => void;
  onPickFrame?: (videoId: string, frame: number) => void;
  csvRows?: CsvRow[];
  queryType?: string;
  onRemoveRow?: (i: number) => void;
};

function rowSummary(row: CsvRow): string {
  const frames = Object.entries(row)
    .filter(([k]) => k.startsWith("frame"))
    .map(([, v]) => v)
    .filter((v) => v !== "" && v != null);
  const ans = row.answer ? ` · "${row.answer}"` : "";
  return `${row.video_name} · f${frames.join(", ")}${ans}`;
}

export default function VideoInspector({
  videoId,
  initialFrame = 0,
  onClose,
  onPickFrame,
  csvRows = [],
  queryType = "KIS",
  onRemoveRow,
}: Props) {
  const videoRef = useRef<HTMLVideoElement>(null);
  const [info, setInfo] = useState<VideoInfo | null>(null);
  const [frame, setFrame] = useState(initialFrame);
  const [error, setError] = useState<string | null>(null);
  const [jump, setJump] = useState(String(initialFrame));
  const [toast, setToast] = useState<string | null>(null);

  const fps = info?.fps ?? 25;
  const maxFrame = info ? Math.max(0, info.frame_count - 1) : 0;

  useEffect(() => {
    let alive = true;
    getVideoInfo(videoId)
      .then((i) => alive && setInfo(i))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, [videoId]);

  const seekToFrame = useCallback(
    (f: number) => {
      const v = videoRef.current;
      if (!v || !info) return;
      const clamped = Math.max(0, Math.min(maxFrame, Math.round(f)));
      v.currentTime = clamped / fps + 1e-5;
      setFrame(clamped);
      setJump(String(clamped));
    },
    [info, fps, maxFrame]
  );

  const seekedRef = useRef(false);
  const onLoaded = () => {
    if (!seekedRef.current && info) {
      seekedRef.current = true;
      seekToFrame(initialFrame);
    }
  };
  const onTimeUpdate = () => {
    const v = videoRef.current;
    if (v) setFrame(Math.floor(v.currentTime * fps + 1e-4));
  };
  const seekBySeconds = (delta: number) => {
    const v = videoRef.current;
    if (!v) return;
    const dur = Number.isFinite(v.duration) ? v.duration : maxFrame / fps;
    v.currentTime = Math.max(0, Math.min(dur, v.currentTime + delta));
    onTimeUpdate();
  };

  const pick = () => {
    onPickFrame?.(videoId, frame);
    setToast(`✓ Đã thêm ${videoId} · frame ${frame}`);
    window.setTimeout(() => setToast(null), 1600);
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4" onClick={onClose}>
      <div
        className="max-h-[92vh] w-full max-w-6xl overflow-auto rounded-xl bg-neutral-900 p-4 text-neutral-100 shadow-2xl"
        onClick={(e) => e.stopPropagation()}
      >
        <div className="mb-2 flex items-center justify-between">
          <h3 className="font-mono text-lg font-bold">
            {videoId} · FRAME {frame}
          </h3>
          <button className="rounded bg-neutral-700 px-3 py-1 text-sm hover:bg-neutral-600" onClick={onClose}>
            Đóng ✕
          </button>
        </div>

        <div className="flex flex-col gap-4 lg:flex-row">
          {/* LEFT: player */}
          <div className="min-w-0 flex-1">
            {error ? (
              <p className="text-red-400">Lỗi tải video: {error}</p>
            ) : (
              <div className="relative">
                <video
                  ref={videoRef}
                  src={videoSrc(videoId)}
                  preload="metadata"
                  controls
                  className="max-h-[55vh] w-full rounded-lg bg-black"
                  onLoadedMetadata={onLoaded}
                  onTimeUpdate={onTimeUpdate}
                  onSeeked={onTimeUpdate}
                />
                <div className="pointer-events-none absolute left-2 top-2 rounded bg-black/80 px-2 py-1 font-mono text-sm font-bold">
                  {videoId} | FRAME {frame}
                </div>
                {toast && (
                  <div className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded bg-green-600 px-3 py-1.5 text-sm font-semibold shadow-lg">
                    {toast}
                  </div>
                )}
              </div>
            )}

            <div className="mt-3 flex flex-wrap items-center gap-2">
              <button className="rounded bg-neutral-700 px-2 py-1 text-sm hover:bg-neutral-600" onClick={() => seekBySeconds(-5)} title="Lùi 5 giây">
                ↶ 5s
              </button>
              <button className="rounded bg-neutral-700 px-2 py-1 text-sm hover:bg-neutral-600" onClick={() => seekBySeconds(5)} title="Tới 5 giây">
                5s ↷
              </button>
              <input
                type="range"
                min={0}
                max={maxFrame}
                value={frame}
                onChange={(e) => seekToFrame(Number(e.target.value))}
                className="min-w-[140px] flex-1"
              />
              <div className="flex items-center gap-1">
                <span className="text-sm text-neutral-400">Frame</span>
                <input
                  type="number"
                  min={0}
                  max={maxFrame}
                  value={jump}
                  onChange={(e) => setJump(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && seekToFrame(Number(jump))}
                  className="w-24 rounded border border-neutral-600 bg-neutral-800 px-2 py-1 text-sm"
                />
                <button className="rounded bg-neutral-700 px-2 py-1 text-sm hover:bg-neutral-600" onClick={() => seekToFrame(Number(jump))}>
                  Go
                </button>
              </div>
              {onPickFrame && (
                <button className="rounded bg-brand px-3 py-1.5 text-sm font-semibold text-brand-fg hover:opacity-90" onClick={pick}>
                  + Dùng frame này cho CSV
                </button>
              )}
            </div>
            {info && (
              <p className="mt-2 text-xs text-neutral-400">
                {info.fps.toFixed(2)} FPS · {info.frame_count.toLocaleString()} frames · {info.width}×{info.height}
              </p>
            )}
          </div>

          {/* RIGHT: live CSV panel */}
          <aside className="w-full shrink-0 rounded-lg bg-neutral-800 p-3 lg:w-72">
            <p className="mb-2 text-sm font-semibold">
              File nộp ({queryType}) · {csvRows.length} dòng
            </p>
            {csvRows.length === 0 ? (
              <p className="text-xs text-neutral-400">
                Chưa có dòng nào. Bấm “+ Dùng frame này cho CSV” để thêm.
              </p>
            ) : (
              <ul className="space-y-1">
                {csvRows.map((r, i) => (
                  <li key={i} className="flex items-center justify-between gap-2 rounded bg-neutral-900/60 px-2 py-1 font-mono text-[11px]">
                    <span className="truncate">
                      {i + 1}. {rowSummary(r)}
                    </span>
                    {onRemoveRow && (
                      <button className="shrink-0 text-neutral-400 hover:text-red-400" onClick={() => onRemoveRow(i)}>
                        ✕
                      </button>
                    )}
                  </li>
                ))}
              </ul>
            )}
            <p className="mt-2 text-[11px] text-neutral-500">
              Cuộn xuống trang để đặt tên & tải CSV.
            </p>
          </aside>
        </div>
      </div>
    </div>
  );
}
