#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import mimetypes
import os
import posixpath
from functools import partial
from http import HTTPStatus
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


DEFAULT_FRAMES_CANDIDATES = [
    "/tmp/corex_simple_final_accept_90f",
    "/tmp/corex_simple_fixrot_90f",
    "/tmp/corex_simple_final_plan_90f",
]


def _pick_default_frames_dir() -> str | None:
    env_dir = os.environ.get("OBJ_VIEWER_FRAMES_DIR", "").strip()
    if env_dir:
        p = Path(env_dir)
        if p.is_dir() and any(p.glob("scene_surface_*.obj")):
            return str(p)
    for d in DEFAULT_FRAMES_CANDIDATES:
        p = Path(d)
        if p.is_dir() and any(p.glob("scene_surface_*.obj")):
            return str(p)
    return None


def build_manifest(frames_dir: Path, fps: int) -> dict:
    frames = sorted(p.name for p in frames_dir.glob("scene_surface_*.obj"))
    if not frames:
        raise SystemExit(f"No OBJ frames found in {frames_dir}")

    return {
        "title": frames_dir.name,
        "frameCount": len(frames),
        "frames": frames,
        "fps": fps,
    }


class ViewerHandler(SimpleHTTPRequestHandler):
    def __init__(self, *args, viewer_root: Path, frames_dir: Path, manifest: dict, **kwargs):
        self.viewer_root = viewer_root
        self.frames_dir = frames_dir
        self.manifest = manifest
        super().__init__(*args, directory=str(viewer_root), **kwargs)

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/manifest":
            payload = json.dumps(self.manifest).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return

        if path.startswith("/frames/"):
            rel_path = unquote(path[len("/frames/"):])
            safe_rel = posixpath.normpath("/" + rel_path).lstrip("/")
            file_path = (self.frames_dir / safe_rel).resolve()
            frames_root = self.frames_dir.resolve()
            if not str(file_path).startswith(str(frames_root)) or not file_path.is_file():
                self.send_error(HTTPStatus.NOT_FOUND, "Frame not found")
                return
            self._serve_file(file_path)
            return

        if path in ("/", "/index.html"):
            self._serve_file(self.viewer_root / "index.html")
            return

        super().do_GET()

    def _serve_file(self, file_path: Path):
        data = file_path.read_bytes()
        mime_type, _ = mimetypes.guess_type(str(file_path))
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", mime_type or "application/octet-stream")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def parse_args() -> argparse.Namespace:
    default_frames = _pick_default_frames_dir()
    parser = argparse.ArgumentParser(description="Serve a lightweight OBJ sequence viewer.")
    parser.add_argument(
        "--frames-dir",
        default=default_frames,
        help=(
            "Directory containing scene_surface_XXXX.obj files. "
            "If omitted, auto-detect from known run directories, or use OBJ_VIEWER_FRAMES_DIR."
        ),
    )
    parser.add_argument("--host", default="0.0.0.0", help="Bind host.")
    parser.add_argument("--port", type=int, default=18080, help="Bind port.")
    parser.add_argument("--fps", type=int, default=12, help="Default playback FPS.")
    return parser.parse_args()


def main():
    args = parse_args()
    viewer_root = Path(__file__).resolve().parent
    if not args.frames_dir:
        raise SystemExit(
            "No frames directory selected. Pass --frames-dir or set OBJ_VIEWER_FRAMES_DIR."
        )

    frames_dir = Path(args.frames_dir).resolve()

    if not frames_dir.is_dir():
        raise SystemExit(f"Frames directory does not exist: {frames_dir}")

    manifest = build_manifest(frames_dir, args.fps)
    handler = partial(
        ViewerHandler,
        viewer_root=viewer_root,
        frames_dir=frames_dir,
        manifest=manifest,
    )
    server = ThreadingHTTPServer((args.host, args.port), handler)

    print(f"Serving {frames_dir} on http://{args.host}:{args.port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
