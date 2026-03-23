"""Generates a QR code pointing to the BioScan AI mobile capture page.

Usage::

    python frontend/qr_generator.py https://abc123.ngrok.io

The QR code image is written to frontend/static/qr_mobile.png and is served
by FastAPI at /static/static/qr_mobile.png (because frontend/ is mounted at
/static).  The desktop UI loads it from that path automatically.

Requirements::

    pip install qrcode[pil]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import qrcode
except ImportError:
    print(
        "ERROR: 'qrcode' is not installed.\n"
        "Install it with:  pip install qrcode[pil]",
        file=sys.stderr,
    )
    sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate a QR code linking to the BioScan AI mobile page.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Example:\n"
            "  python frontend/qr_generator.py https://abc123.ngrok.io\n\n"
            "For local development, first expose the server with:\n"
            "  ngrok http 8000\n"
            "then pass the HTTPS ngrok URL as base_url."
        ),
    )
    parser.add_argument(
        "base_url",
        help="Base URL of the running server, e.g. https://abc123.ngrok.io",
    )
    args = parser.parse_args()

    base_url = args.base_url.rstrip("/")
    mobile_url = f"{base_url}/static/mobile.html"

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(mobile_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")

    output_path = Path("frontend/static/qr_mobile.png")
    img.save(output_path)

    print(f"QR code saved → {output_path}")
    print(f"URL encoded   → {mobile_url}")


if __name__ == "__main__":
    main()
