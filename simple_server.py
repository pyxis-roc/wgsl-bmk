#!/usr/bin/env python3

# SPDX-FileCopyrightText: 2024 University of Rochester
#
# SPDX-License-Identifier: MIT

__author__ = "Benjamin Valpey"
__license__ = "MIT"
import http.server
import socketserver
import signal

class CustomHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=ROOT_DIR, **kwargs)

PORT = 5030
ROOT_DIR = "public"

Handler = http.server.SimpleHTTPRequestHandler

Handler.extensions_map = {
    ".manifest": "text/cache-manifest",
    ".wgsl": "text/wgsl",
    ".html": "text/html",
    ".png": "image/png",
    ".jpg": "image/jpg",
    ".svg": "image/svg+xml",
    ".css": "text/css",
    ".js": "application/x-javascript",
    ".ico": "image/vnd.microsoft.icon",
    "": "application/octet-stream",  # Default
}

global httpd
def signal_handler(sig, frame):
    print('Interrupt received, shutting down the server...')
    if httpd is not None:
        httpd.server_close()
        import sys
        sys.exit(0)


if __name__ == '__main__':
    signal.signal(signal.SIGINT, signal_handler)
    with socketserver.TCPServer(("", PORT), CustomHandler) as httpd:
        httpd.allow_reuse_address = True
        httpd.allow_reuse_port = True
        print("serving at port", PORT)
        httpd.serve_forever()


