#!/usr/bin/env python3
"""
view_timelines.py

Companion to hardcoded_text_timelines.py.

Opens an interactive HTML viewer with your annotation data pre-loaded.
Color-coded bars: green = perfect, amber = yes, dark = no.

Usage:
    python view_timelines.py

By default it reads the SAME input file your bar-builder uses
(combo_data/combined.txt). Change INPUT_PATH below if you want
to point it at bars_output.txt or something else — it auto-detects
both formats (raw annotations OR pre-built [   o O o ] bars).
"""

import os
import sys
import json
import tempfile
import webbrowser
from pathlib import Path

# ---------- Settings ----------
# Point this at either:
#   - the raw annotations file (same as your bar builder uses), OR
#   - bars_output.txt (the pre-built bars)
# The viewer auto-detects which format each line is in.
INPUT_PATH = "combo_data/combined.txt"

# Path to the standalone viewer HTML (sits next to this script by default)
VIEWER_HTML = Path(__file__).parent / "timeline_viewer.html"
# ------------------------------


HTML_TEMPLATE_FALLBACK_MSG = """
Could not find timeline_viewer.html next to this script.
Make sure timeline_viewer.html is in the same folder as view_timelines.py.
"""


def main():
    if not VIEWER_HTML.is_file():
        print(HTML_TEMPLATE_FALLBACK_MSG, file=sys.stderr)
        sys.exit(1)

    if not os.path.isfile(INPUT_PATH):
        print(f"Input file not found: {INPUT_PATH}", file=sys.stderr)
        print("Edit INPUT_PATH at the top of this script.", file=sys.stderr)
        sys.exit(1)

    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        data = f.read()

    html = VIEWER_HTML.read_text(encoding="utf-8")

    # Inject the file contents as a JS variable the viewer will pick up
    # on load instead of the built-in SAMPLE.
    injection = (
        "<script>window.__PRELOAD_DATA__ = "
        + json.dumps(data)
        + ";</script>\n</body>"
    )
    html = html.replace("</body>", injection, 1)

    # Also patch the auto-load so it prefers preloaded data when present.
    patch_find = "document.getElementById('input').value = SAMPLE;\n    document.getElementById('render').click();\n  });"
    patch_repl = (
        "const pre = window.__PRELOAD_DATA__;\n"
        "    document.getElementById('input').value = (typeof pre === 'string' && pre.trim()) ? pre : SAMPLE;\n"
        "    document.getElementById('render').click();\n"
        "  });"
    )
    html = html.replace(patch_find, patch_repl, 1)

    # Write to a temp file and open it
    tmp = tempfile.NamedTemporaryFile(
        prefix="timeline_view_", suffix=".html", delete=False, mode="w", encoding="utf-8"
    )
    tmp.write(html)
    tmp.close()

    url = "file://" + os.path.abspath(tmp.name)
    print(f"Opening: {url}")
    webbrowser.open(url)


if __name__ == "__main__":
    main()