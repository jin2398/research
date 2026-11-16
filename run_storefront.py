"""Launch the pixel-identical browser storefront without Streamlit.
Run: python run_storefront.py
"""

import webbrowser
from functools import partial
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent
port = 8503
url = f"http://localhost:{port}/codisay_preview.html"
print(f"RE:SEARCH 화면: {url} (종료: Ctrl+C)")
webbrowser.open(url)
ThreadingHTTPServer(
    ("127.0.0.1", port), partial(SimpleHTTPRequestHandler, directory=str(ROOT))
).serve_forever()
