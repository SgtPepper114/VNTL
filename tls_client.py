"""TLS verification policy for API clients, without changing global SSL state."""

from __future__ import annotations

import os
import ssl
import sys


def tls_verification() -> ssl.SSLContext | bool:
    """Use Windows trust roots unless HTTPX has an explicit CA override.

    Returning True lets HTTPX preserve SSL_CERT_FILE/SSL_CERT_DIR precedence
    and errors. A custom context would otherwise bypass those overrides.
    Proxy selection remains the SDK/HTTPX client's normal trust_env behavior.
    """
    if os.environ.get("SSL_CERT_FILE") or os.environ.get("SSL_CERT_DIR"):
        return True
    if sys.platform == "win32":
        import truststore

        return truststore.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    return True
