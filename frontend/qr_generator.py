"""Generates a QR code pointing to the BioScan AI mobile capture page.

Usage::

    python frontend/qr_generator.py https://abc123.ngrok.io

The QR code image is written to frontend/static/qr_mobile.png and is served
by FastAPI at /static/qr_mobile.png.

Requirements::

    pip install qrcode[pil]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

try:
    import qrcode
except ImportError as _qr_missing:
    qrcode = None  # type: ignore[assignment]
    _QR_MISSING_ERROR = _qr_missing
else:
    _QR_MISSING_ERROR = None

# Saved directly under frontend/ so FastAPI serves it at /static/qr_mobile.png
# (frontend/ is mounted at /static, so frontend/qr_mobile.png → /static/qr_mobile.png).
_OUTPUT_PATH = Path(__file__).resolve().parent / "qr_mobile.png"


def generate_qr(base_url: str) -> None:
    """Generate a QR code linking to the mobile page and save it to
    frontend/qr_mobile.png (served by FastAPI at /static/qr_mobile.png).

    Args:
        base_url: Base URL of the running server, e.g. https://abc123.ngrok.io
                  or http://127.0.0.1:8000 for local development.
    """
    if _QR_MISSING_ERROR is not None:
        raise ImportError(
            "'qrcode' is not installed. Install it with: pip install qrcode[pil]"
        ) from _QR_MISSING_ERROR

    mobile_url = f"{base_url.rstrip('/')}/static/mobile.html"

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_M,
        box_size=10,
        border=4,
    )
    qr.add_data(mobile_url)
    qr.make(fit=True)

    img = qr.make_image(fill_color="black", back_color="white")
    _OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    img.save(_OUTPUT_PATH)


def main() -> None:  # pragma: no cover
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

    if _QR_MISSING_ERROR is not None:
        print(
            "ERROR: 'qrcode' is not installed.\n"
            "Install it with:  pip install qrcode[pil]",
            file=sys.stderr,
        )
        sys.exit(1)

    generate_qr(args.base_url)

    mobile_url = f"{args.base_url.rstrip('/')}/static/mobile.html"
    print(f"QR code saved → {_OUTPUT_PATH}")
    print(f"URL encoded   → {mobile_url}")


if __name__ == "__main__":
    main()
