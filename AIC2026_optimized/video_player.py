from __future__ import annotations

import hashlib
import html
import json
import threading
from functools import lru_cache
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import cv2
import streamlit.components.v1 as components
from huggingface_hub import hf_hub_download

from config import HF_VIDEO_REPOS, LOCAL_VIDEO_ROOTS

VIDEO_ASSET_REGISTRY: dict[str, dict] = {}
_VIDEO_HTTP_SERVER: ThreadingHTTPServer | None = None
_VIDEO_HTTP_PORT: int | None = None
_VIDEO_HTTP_LOCK = threading.Lock()


def normalize_video_name(video_name: str) -> str:
    name = str(video_name or "").strip()
    if name.lower().endswith(".mp4"):
        name = name[:-4]
    return name


def infer_source_folder(video_id: str) -> str:
    return f"Videos_{video_id.split('_', 1)[0]}"


def candidate_video_paths(video_id: str) -> list[str]:
    source_folder = infer_source_folder(video_id)
    prefix = video_id.split("_", 1)[0]
    candidates = [
        f"{source_folder}/{video_id}.mp4",
        f"videos/{source_folder}/{video_id}.mp4",
        f"Videos/{source_folder}/{video_id}.mp4",
        f"videos/{prefix}/{video_id}.mp4",
        f"Videos/{prefix}/{video_id}.mp4",
        f"{video_id}.mp4",
    ]
    return list(dict.fromkeys(candidates))


def find_local_video(video_id: str) -> Path | None:
    filename = f"{video_id}.mp4"
    source_folder = infer_source_folder(video_id)
    for root in LOCAL_VIDEO_ROOTS:
        for path in (root / source_folder / filename, root / filename):
            if path.exists() and path.is_file():
                return path.resolve()
    return None


@lru_cache(maxsize=128)
def download_video_from_hf_cached(video_id: str, token: str, exact_hf_path: str = "", preferred_repo_id: str = "") -> tuple[str, str, str]:
    video_id = normalize_video_name(video_id)
    if not video_id:
        raise ValueError("Tên video đang trống.")

    local = find_local_video(video_id)
    if local is not None:
        return str(local), "LOCAL", str(local)

    repo_candidates = list(HF_VIDEO_REPOS)
    if preferred_repo_id:
        repo_candidates.sort(key=lambda x: 0 if x[0] == preferred_repo_id else 1)
    paths = [exact_hf_path.strip()] if exact_hf_path.strip() else candidate_video_paths(video_id)

    errors: list[str] = []
    for repo_id, repo_type in repo_candidates:
        for path_in_repo in paths:
            try:
                local_path = hf_hub_download(
                    repo_id=repo_id,
                    filename=path_in_repo,
                    repo_type=repo_type,
                    token=token,
                )
                return local_path, repo_id, path_in_repo
            except Exception as exc:
                errors.append(f"{repo_id} :: {path_in_repo} -> {type(exc).__name__}")

    raise FileNotFoundError(
        "Không tìm thấy video. Đã thử:\n" + "\n".join(errors[:12])
    )


@lru_cache(maxsize=256)
def get_video_info(video_path: str) -> dict:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"OpenCV không mở được video: {video_path}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    if fps <= 0:
        fps = 25.0
    duration = frame_count / fps if frame_count > 0 else 0.0
    return {"fps": fps, "frame_count": frame_count, "duration": duration, "width": width, "height": height}


class _VideoAssetHandler(BaseHTTPRequestHandler):
    server_version = "AICVideoHTTP/0.2"

    def log_message(self, format: str, *args) -> None:  # noqa: A003
        return

    def do_GET(self) -> None:  # noqa: N802
        parsed = urlparse(self.path)
        if parsed.path != "/video":
            self.send_error(404, "Unsupported path")
            return
        asset_id = parse_qs(parsed.query).get("id", [""])[0]
        asset = VIDEO_ASSET_REGISTRY.get(asset_id)
        if not asset:
            self.send_error(404, "Unknown video asset")
            return
        self._serve_video_file(Path(asset["path"]))

    def _serve_video_file(self, file_path: Path) -> None:
        if not file_path.exists() or not file_path.is_file():
            self.send_error(404, "Video file not found")
            return

        file_size = file_path.stat().st_size
        start, end, status_code = 0, file_size - 1, 200
        range_header = self.headers.get("Range")

        if range_header and range_header.startswith("bytes="):
            try:
                value = range_header.split("=", 1)[1].split(",", 1)[0]
                start_str, end_str = value.split("-", 1)
                if not start_str and end_str:
                    suffix_len = min(int(end_str), file_size)
                    start, end = file_size - suffix_len, file_size - 1
                else:
                    start = int(start_str) if start_str else 0
                    end = int(end_str) if end_str else file_size - 1
                    end = min(end, file_size - 1)
                if start < 0 or start >= file_size or start > end:
                    self.send_response(416)
                    self.send_header("Content-Range", f"bytes */{file_size}")
                    self.end_headers()
                    return
                status_code = 206
            except Exception:
                start, end, status_code = 0, file_size - 1, 200

        chunk_len = end - start + 1
        self.send_response(status_code)
        self.send_header("Content-Type", "video/mp4")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Content-Length", str(chunk_len))
        self.send_header("Cache-Control", "public, max-age=3600")
        if status_code == 206:
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
        self.end_headers()

        try:
            with file_path.open("rb") as f:
                f.seek(start)
                remaining = chunk_len
                while remaining > 0:
                    data = f.read(min(1024 * 1024, remaining))
                    if not data:
                        break
                    self.wfile.write(data)
                    remaining -= len(data)
        except (BrokenPipeError, ConnectionResetError):
            return


def ensure_video_http_server() -> int:
    global _VIDEO_HTTP_SERVER, _VIDEO_HTTP_PORT
    with _VIDEO_HTTP_LOCK:
        if _VIDEO_HTTP_SERVER is not None and _VIDEO_HTTP_PORT is not None:
            return _VIDEO_HTTP_PORT
        server = ThreadingHTTPServer(("0.0.0.0", 0), _VideoAssetHandler)
        port = int(server.server_address[1])
        threading.Thread(target=server.serve_forever, daemon=True).start()
        _VIDEO_HTTP_SERVER, _VIDEO_HTTP_PORT = server, port
        return port


def register_video_asset(video_path: str, video_id: str, info: dict) -> str:
    raw = f"{Path(video_path).resolve()}::{video_id}"
    asset_id = hashlib.sha1(raw.encode("utf-8")).hexdigest()[:16]
    VIDEO_ASSET_REGISTRY[asset_id] = {
        "path": str(Path(video_path).resolve()),
        "video_id": str(video_id),
        "fps": float(info["fps"]),
        "frame_count": int(info["frame_count"]),
    }
    return asset_id


def render_custom_video_browser(video_path: str, video_id: str, info: dict, initial_frame: int = 0) -> None:
    port = ensure_video_http_server()
    asset_id = register_video_asset(video_path, video_id, info)
    fps = float(info["fps"])
    frame_count = int(info["frame_count"])
    video_width = max(1, int(info.get("width", 16) or 16))
    video_height = max(1, int(info.get("height", 9) or 9))
    max_frame = max(0, frame_count - 1)
    initial_frame = max(0, min(int(initial_frame), max_frame))
    player_max_height_px = 648
    player_max_width_px = max(320, int(round(player_max_height_px * (video_width / video_height))))

    cfg = json.dumps({
        "videoId": str(video_id), "fps": fps, "frameCount": frame_count,
        "maxFrame": max_frame, "initialFrame": initial_frame,
        "videoWidth": video_width, "videoHeight": video_height,
        "playerMaxWidth": player_max_width_px, "port": int(port), "assetId": asset_id,
    }, ensure_ascii=False)

    # Hover preview uses a second browser <video>, not OpenCV frame extraction.
    # Main video seeks only when the scrub is released, avoiding dozens of seeks/sec.
    html_block = f"""
<div id="aic-player-root"></div>
<script>
const CFG = {cfg};
const host = window.location.hostname || '127.0.0.1';
const videoUrl = `http://${{host}}:${{CFG.port}}/video?id=${{encodeURIComponent(CFG.assetId)}}`;
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const frameFromTime = s => clamp(Math.floor(Math.max(0,s)*CFG.fps + 0.0001), 0, CFG.maxFrame);
function fmt(s) {{ const x=Math.max(0,Math.floor(Number(s)||0)), ss=String(x%60).padStart(2,'0'), mm=String(Math.floor(x/60)%60).padStart(2,'0'), h=Math.floor(x/3600); return h?`${{String(h).padStart(2,'0')}}:${{mm}}:${{ss}}`:`${{mm}}:${{ss}}`; }}
const root=document.getElementById('aic-player-root');
root.innerHTML=`
<style>
*{{box-sizing:border-box}} .wrap{{width:100%;display:flex;justify-content:center;font-family:system-ui,sans-serif}}
.shell{{position:relative;width:min(100%,${{CFG.playerMaxWidth}}px);aspect-ratio:${{CFG.videoWidth}}/${{CFG.videoHeight}};background:#000;border-radius:12px;overflow:hidden;box-shadow:0 8px 26px rgba(0,0,0,.22)}}
.shell>video.main{{display:block;width:100%;height:100%;object-fit:contain;background:#000;cursor:pointer}}
.overlay{{position:absolute;top:8px;left:8px;z-index:8;background:rgba(0,0,0,.84);color:white;font:800 15px ui-monospace,monospace;padding:6px 10px;border-radius:3px;pointer-events:none}}
.controls{{position:absolute;left:0;right:0;bottom:0;z-index:12;display:flex;align-items:center;gap:8px;padding:30px 10px 9px;background:linear-gradient(to top,rgba(0,0,0,.82),transparent)}}
.btn,.skip,.go{{border:0;border-radius:6px;background:rgba(0,0,0,.35);color:#fff;cursor:pointer;height:30px}} .btn{{width:30px}} .skip{{min-width:46px;font-size:11px;font-weight:700}} .go{{height:28px;font-size:11px;font-weight:700}}
.scrub{{position:relative;flex:1;min-width:100px;display:flex;align-items:center}} .scrub input{{width:100%;margin:0;cursor:pointer;accent-color:white}}
.time{{min-width:92px;color:#fff;font:650 11px ui-monospace,monospace;text-align:center;white-space:nowrap}} .jump{{display:flex;align-items:center;gap:4px;color:white;font-size:11px;font-weight:650}}
.frame{{width:78px;height:28px;border:1px solid rgba(255,255,255,.45);border-radius:5px;background:rgba(0,0,0,.58);color:#fff;padding:2px 6px;font:12px ui-monospace,monospace}}
.preview{{display:none;position:absolute;z-index:20;width:210px;padding:5px;background:rgba(0,0,0,.94);border-radius:7px;box-shadow:0 5px 20px rgba(0,0,0,.5);pointer-events:none}}
.preview video{{width:100%;height:auto;display:block;border-radius:4px;background:#111}} .preview .meta{{margin-top:4px;color:#fff;font:700 11px ui-monospace,monospace;text-align:center;white-space:nowrap}}
@media(max-width:800px){{.time,.jump span{{display:none}}.frame{{width:68px}}.preview{{width:170px}}}}
</style>
<div class="wrap"><div class="shell" id="shell">
<video class="main" id="video" preload="metadata" src="${{videoUrl}}"></video><div class="overlay" id="overlay"></div>
<div class="preview" id="preview"><video id="previewVideo" preload="metadata" muted src="${{videoUrl}}"></video><div class="meta" id="previewMeta"></div></div>
<div class="controls" id="controls"><button class="btn" id="play">▶</button><button class="skip" id="back">↶ 5s</button><button class="skip" id="forward">5s ↷</button>
<div class="scrub"><input id="scrub" type="range" min="0" max="${{CFG.maxFrame}}" step="1" value="${{CFG.initialFrame}}"></div><div class="time" id="time">00:00 / 00:00</div>
<div class="jump"><span>Frame</span><input class="frame" id="frameInput" type="number" min="0" max="${{CFG.maxFrame}}" value="${{CFG.initialFrame}}"><button class="go" id="go">Go</button></div>
<button class="btn" id="mute">🔊</button><button class="btn" id="full">⛶</button></div></div></div>`;
const shell=document.getElementById('shell'), video=document.getElementById('video'), pv=document.getElementById('previewVideo'), overlay=document.getElementById('overlay'), preview=document.getElementById('preview'), previewMeta=document.getElementById('previewMeta'), controls=document.getElementById('controls'), play=document.getElementById('play'), back=document.getElementById('back'), forward=document.getElementById('forward'), scrub=document.getElementById('scrub'), time=document.getElementById('time'), frameInput=document.getElementById('frameInput'), go=document.getElementById('go'), mute=document.getElementById('mute'), full=document.getElementById('full');
let isScrubbing=false,lastScrub=CFG.initialFrame,previewTimer=null,pendingPreviewTime=null;
function updateUI(frameOverride=null){{const f=frameOverride===null?frameFromTime(video.currentTime):clamp(Math.round(frameOverride),0,CFG.maxFrame);const shownTime=frameOverride===null?video.currentTime:f/CFG.fps;overlay.textContent=`${{CFG.videoId}} | FRAME ${{f}}`;if(!isScrubbing)scrub.value=String(f);if(document.activeElement!==frameInput)frameInput.value=String(f);time.textContent=`${{fmt(shownTime)}} / ${{fmt(video.duration)}}`;play.textContent=video.paused?'▶':'❚❚';mute.textContent=(video.muted||video.volume===0)?'🔇':'🔊'}}
function seekFrame(raw){{let f=Number(raw);if(!Number.isFinite(f))return;f=clamp(Math.round(f),0,CFG.maxFrame);video.currentTime=f/CFG.fps+0.00001;scrub.value=String(f);frameInput.value=String(f);updateUI(f)}}
function seekSec(d){{if(!Number.isFinite(video.duration))return;video.currentTime=clamp(video.currentTime+d,0,video.duration);updateUI()}}
function toggle(){{video.paused?video.play().catch(()=>{{}}):video.pause()}}
function requestPreview(f){{const t=clamp(f/CFG.fps,0,Math.max(0,video.duration||0));pendingPreviewTime=t;if(previewTimer)clearTimeout(previewTimer);previewTimer=setTimeout(()=>{{if(pv.readyState>=1)pv.currentTime=pendingPreviewTime;else pv.addEventListener('loadedmetadata',()=>{{pv.currentTime=pendingPreviewTime}},{{once:true}})}},140)}}
function previewAt(e){{const r=scrub.getBoundingClientRect(),ratio=r.width?clamp((e.clientX-r.left)/r.width,0,1):0,f=Math.round(ratio*CFG.maxFrame),sr=shell.getBoundingClientRect(),w=preview.offsetWidth||210;let left=e.clientX-sr.left-w/2;left=clamp(left,5,Math.max(5,sr.width-w-5));preview.style.left=`${{left}}px`;preview.style.bottom=`${{Math.max(54,controls.offsetHeight+8)}}px`;preview.style.display='block';previewMeta.textContent=`${{CFG.videoId}} | FRAME ${{f}} · ${{fmt(f/CFG.fps)}}`;requestPreview(f)}}
video.addEventListener('loadedmetadata',()=>{{seekFrame(CFG.initialFrame);updateUI(CFG.initialFrame)}});video.addEventListener('click',toggle);video.addEventListener('play',()=>updateUI());video.addEventListener('pause',()=>updateUI());video.addEventListener('seeked',()=>updateUI());video.addEventListener('timeupdate',()=>{{if(!isScrubbing)updateUI()}});
if(video.requestVideoFrameCallback){{const loop=()=>video.requestVideoFrameCallback((_,m)=>{{if(!isScrubbing)updateUI(frameFromTime(m.mediaTime));loop()}});loop()}}
play.onclick=e=>{{e.preventDefault();toggle()}};back.onclick=e=>{{e.preventDefault();seekSec(-5)}};forward.onclick=e=>{{e.preventDefault();seekSec(5)}};
scrub.addEventListener('input',e=>{{isScrubbing=true;lastScrub=Number(e.target.value||0);updateUI(lastScrub)}});scrub.addEventListener('change',e=>{{lastScrub=Number(e.target.value||0);seekFrame(lastScrub);isScrubbing=false;setTimeout(()=>preview.style.display='none',100)}});scrub.addEventListener('mousemove',previewAt);scrub.addEventListener('mouseenter',previewAt);scrub.addEventListener('mouseleave',()=>{{if(!isScrubbing)preview.style.display='none'}});
go.onclick=()=>seekFrame(frameInput.value);frameInput.addEventListener('keydown',e=>{{if(e.key==='Enter'){{e.preventDefault();seekFrame(frameInput.value);frameInput.blur()}}}});mute.onclick=()=>{{video.muted=!video.muted;updateUI()}};full.onclick=()=>{{document.fullscreenElement?document.exitFullscreen().catch(()=>{{}}):shell.requestFullscreen?.().catch(()=>{{}})}};updateUI(CFG.initialFrame);
</script>
"""
    components.html(html_block, height=690, scrolling=False)


def format_seconds(seconds: float) -> str:
    total_ms = max(0, int(round(seconds * 1000)))
    ms = total_ms % 1000
    total_s = total_ms // 1000
    s = total_s % 60
    total_m = total_s // 60
    m = total_m % 60
    h = total_m // 60
    return f"{h:02d}:{m:02d}:{s:02d}.{ms:03d}" if h else f"{m:02d}:{s:02d}.{ms:03d}"
