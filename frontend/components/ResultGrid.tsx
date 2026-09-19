"use client";

import { imageSrc, type Hit } from "@/lib/api";

function scoreOf(h: Hit): string {
  if (typeof h.rrf_score === "number") return `rrf ${h.rrf_score.toFixed(4)}`;
  if (typeof h.score === "number") return h.score.toFixed(3);
  return "";
}

export function hitKey(h: Hit, i: number): string {
  return `${h.video_id}#${h.frame_id}#${i}`;
}

type Props = {
  results: Hit[];
  onOpen: (h: Hit) => void;
  selectedKeys: Set<string>;
  onToggle: (key: string) => void;
};

export default function ResultGrid({ results, onOpen, selectedKeys, onToggle }: Props) {
  if (results.length === 0) return null;
  return (
    <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5">
      {results.map((h, i) => {
        const key = hitKey(h, i);
        const checked = selectedKeys.has(key);
        return (
          <div
            key={key}
            className={`group relative overflow-hidden rounded-lg border bg-white text-left shadow-sm transition hover:shadow-md dark:bg-neutral-900 ${
              checked ? "border-brand ring-2 ring-brand" : "border-neutral-200 dark:border-neutral-800"
            }`}
          >
            <label
              className="absolute right-1 top-1 z-10 flex cursor-pointer items-center rounded bg-black/60 px-1 py-0.5"
              onClick={(e) => e.stopPropagation()}
            >
              <input
                type="checkbox"
                checked={checked}
                onChange={() => onToggle(key)}
                className="h-4 w-4 cursor-pointer accent-indigo-500"
              />
            </label>

            <button onClick={() => onOpen(h)} className="block w-full text-left">
              <div className="relative aspect-video bg-neutral-200 dark:bg-neutral-800">
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img
                  src={imageSrc(h)}
                  alt={`${h.video_id} #${h.frame_id}`}
                  loading="lazy"
                  className="h-full w-full object-cover"
                  onError={(e) => (e.currentTarget.style.visibility = "hidden")}
                />
                <span className="absolute left-1 top-1 rounded bg-black/70 px-1.5 py-0.5 font-mono text-[11px] font-bold text-white">
                  #{h.rank ?? i + 1}
                </span>
              </div>
              <div className="p-2">
                <p className="truncate font-mono text-xs font-semibold">
                  {h.video_id} · f{h.frame_id}
                </p>
                <p className="text-[11px] text-neutral-500">{scoreOf(h)}</p>
                {h.sources && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {h.sources.map((s) => (
                      <span key={s} className="rounded bg-brand/10 px-1 py-0.5 text-[10px] font-medium text-brand">
                        {s.replace("meta:", "")}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </button>
          </div>
        );
      })}
    </div>
  );
}
