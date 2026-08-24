#!/usr/bin/env python3
"""Export 3D screenshots using the Streamlit Bulleen Three.js viewer.

This script reuses ``streamlit_bulleen/app_interface/layout_viewer_3d.html`` so
the screenshots match the interactive viewer rather than a simplified Matplotlib
approximation.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import tempfile
import time
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
VIEWER_TEMPLATE = REPO_ROOT / "streamlit_bulleen" / "app_interface" / "layout_viewer_3d.html"
DEFAULT_RESULTS_DIR = PROJECT_DIR / "results" / "cexo_bulleen_15000_full_fg"
THREE_VENDOR = PROJECT_DIR / "vendor" / "three"

CHROME_CANDIDATES = [
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files\Microsoft\Edge\Application\msedge.exe"),
    Path(r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"),
]


def find_browser() -> Path:
    for path in CHROME_CANDIDATES:
        if path.exists():
            return path
    raise SystemExit("Could not find Chrome or Edge for headless screenshot export.")


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def selected_layout_files(results_dir: Path, limit: int) -> list[tuple[str, Path]]:
    table_path = results_dir / "paper_analysis" / "table_showcase_layouts.csv"
    if table_path.exists():
        rows = list(csv.DictReader(table_path.open("r", encoding="utf-8")))
        selected = []
        for row in rows:
            path = results_dir / (row.get("json_file") or f"{row['layout_id']}.json")
            if path.exists():
                selected.append((row.get("showcase_role") or path.stem, path))
            if len(selected) >= limit:
                return selected

    files = sorted(results_dir.glob("cslpelite_layout_*.json"))[:limit]
    return [(path.stem, path) for path in files]


def make_standalone_html(layout: dict, output_html: Path, clean: bool, camera: str) -> None:
    template = VIEWER_TEMPLATE.read_text(encoding="utf-8")
    three_module = (THREE_VENDOR / "three.module.js").resolve().as_uri()
    orbit_controls = (THREE_VENDOR / "OrbitControls.js").resolve().as_uri()
    obj_loader = (THREE_VENDOR / "OBJLoader.js").resolve().as_uri()
    ply_loader = (THREE_VENDOR / "PLYLoader.js").resolve().as_uri()

    template = template.replace(
        '"three": "https://unpkg.com/three@0.160.0/build/three.module.js"',
        f'"three": "{three_module}"',
    )
    template = template.replace(
        'import { OrbitControls } from "https://unpkg.com/three@0.160.0/examples/jsm/controls/OrbitControls.js";',
        f'import {{ OrbitControls }} from "{orbit_controls}";',
    )
    template = template.replace(
        'import { OBJLoader } from "https://unpkg.com/three@0.160.0/examples/jsm/loaders/OBJLoader.js";',
        f'import {{ OBJLoader }} from "{obj_loader}";',
    )
    template = template.replace(
        'import { PLYLoader } from "https://unpkg.com/three@0.160.0/examples/jsm/loaders/PLYLoader.js";',
        f'import {{ PLYLoader }} from "{ply_loader}";',
    )
    template = template.replace(
        "controls.maxPolarAngle = Math.PI / 2;",
        "controls.maxPolarAngle = Math.PI / 2;\n"
        "    window.__bulleenViewer = { camera, controls, scene, renderer };",
    )

    # Match the Streamlit viewer, but allow a clean version for paper use.
    extra_css = ""
    if clean:
        extra_css = """
        <style>
          #controls, #legend { display: none !important; }
          html, body { background: #eef0f4 !important; }
        </style>
        """

    # The original viewer expects Streamlit to post layout_data to the iframe.
    # This standalone wrapper performs the same message post after the module
    # script has initialized its listener.
    camera_script = ""
    if camera == "low":
        camera_script = """
          const viewer = window.__bulleenViewer;
          if (viewer) {
            const target = viewer.controls.target;
            const radius = viewer.camera.position.distanceTo(target);
            viewer.camera.position.set(
              target.x - radius * 0.52,
              Math.max(radius * 0.24, 9),
              target.z + radius * 0.48
            );
            viewer.camera.lookAt(target);
            viewer.controls.update();
          }
        """
    elif camera == "top":
        camera_script = """
          const viewer = window.__bulleenViewer;
          if (viewer) {
            const target = viewer.controls.target;
            const radius = viewer.camera.position.distanceTo(target);
            viewer.camera.position.set(target.x, Math.max(radius * 1.05, 30), target.z + 0.01);
            viewer.camera.lookAt(target);
            viewer.controls.update();
          }
        """

    injection = f"""
    {extra_css}
    <script>
    window.addEventListener('DOMContentLoaded', function() {{
      const layoutData = {json.dumps(layout)};
      setTimeout(function() {{
        window.postMessage({{ type: 'layout_data', data: layoutData }}, '*');
        setTimeout(function() {{
          if (window.resetView) {{
            window.resetView();
          }}
          {camera_script}
          document.body.dataset.ready = '1';
        }}, 1800);
      }}, 600);
    }});
    </script>
    """

    output_html.write_text(template.replace("</body>", injection + "\n</body>"), encoding="utf-8")


def chrome_screenshot(browser: Path, html_path: Path, output_png: Path, width: int, height: int) -> None:
    user_data_dir = Path(tempfile.mkdtemp(prefix="bulleen_chrome_profile_"))
    url = html_path.resolve().as_uri()
    cmd = [
        str(browser),
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--no-first-run",
        "--no-default-browser-check",
        "--allow-file-access-from-files",
        "--hide-scrollbars",
        "--virtual-time-budget=6000",
        f"--user-data-dir={user_data_dir}",
        f"--window-size={width},{height}",
        f"--screenshot={output_png}",
        url,
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=45)
    if result.returncode != 0:
        raise RuntimeError(
            "Chrome screenshot failed\n"
            f"Return code: {result.returncode}\n"
            f"STDOUT:\n{result.stdout}\n"
            f"STDERR:\n{result.stderr}"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    parser.add_argument("--limit", type=int, default=4)
    parser.add_argument("--width", type=int, default=1600)
    parser.add_argument("--height", type=int, default=1100)
    parser.add_argument("--camera", choices=["reset", "low", "top"], default="low")
    parser.add_argument("--include-ui", action="store_true", help="Also export screenshots with controls and legend visible.")
    args = parser.parse_args()

    results_dir = args.results_dir.resolve()
    output_dir = results_dir / "paper_analysis" / "streamlit_3d_exports"
    output_dir.mkdir(parents=True, exist_ok=True)

    browser = find_browser()
    selected = selected_layout_files(results_dir, args.limit)
    if not selected:
        raise SystemExit(f"No layout JSON files found in {results_dir}")

    exported = []
    for index, (role, layout_path) in enumerate(selected, start=1):
        layout = load_json(layout_path)
        safe_role = "".join(ch.lower() if ch.isalnum() else "_" for ch in role).strip("_")

        for clean in [True, False] if args.include_ui else [True]:
            suffix = "clean" if clean else "ui"
            stem = f"{index:02d}_{safe_role}_{layout_path.stem}_{suffix}"
            html_path = output_dir / f"{stem}.html"
            png_path = output_dir / f"{stem}.png"
            make_standalone_html(layout, html_path, clean=clean, camera=args.camera)
            # Chrome's --screenshot does not expose a DOM-ready wait flag, so
            # the HTML deliberately delays setting ready and scene replacement.
            chrome_screenshot(browser, html_path, png_path, args.width, args.height)
            time.sleep(0.2)
            exported.append(png_path)
            print(f"Exported {png_path}")

    manifest = {
        "source_viewer": str(VIEWER_TEMPLATE),
        "results_dir": str(results_dir),
        "browser": str(browser),
        "screenshots": [str(path) for path in exported],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
