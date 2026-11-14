"""Exercise API TLS defaults, explicit CA overrides and real proxy routing."""

import asyncio
import os
import ssl
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import anthropic
import certifi
import openai

from config import Config
from tls_client import tls_verification
from translator import Translator


class TLSClientTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        environment = patch.dict(os.environ, {"NO_PROXY": "*"}, clear=True)
        environment.start()
        self.addCleanup(environment.stop)

    @unittest.skipUnless(sys.platform == "win32", "Windows native trust store")
    async def test_windows_context_checks_certificates_and_hostname_without_global_patch(self):
        import truststore

        original_context = ssl.SSLContext
        context = tls_verification()
        self.assertIsInstance(context, truststore.SSLContext)
        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
        self.assertTrue(context.check_hostname)
        self.assertIs(ssl.SSLContext, original_context)

    async def test_ca_file_overrides_windows_roots_and_takes_precedence_over_directory(self):
        os.environ["SSL_CERT_FILE"] = certifi.where()
        os.environ["SSL_CERT_DIR"] = "nonexistent-ignored-directory"
        for factory in (openai.DefaultAsyncHttpxClient, anthropic.DefaultAsyncHttpxClient):
            with self.subTest(sdk=factory.__module__):
                async with factory(verify=tls_verification()) as client:
                    context = client._transport._pool._ssl_context
                    self.assertIs(type(context), ssl.SSLContext)
                    self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
                    self.assertTrue(context.check_hostname)
                    self.assertEqual(
                        context.get_ca_certs(binary_form=True),
                        ssl.create_default_context(cafile=certifi.where()).get_ca_certs(binary_form=True),
                    )

    async def test_ca_directory_overrides_default_roots(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["SSL_CERT_DIR"] = directory
            for factory in (openai.DefaultAsyncHttpxClient, anthropic.DefaultAsyncHttpxClient):
                with self.subTest(sdk=factory.__module__):
                    async with factory(verify=tls_verification()) as client:
                        context = client._transport._pool._ssl_context
                        self.assertIs(type(context), ssl.SSLContext)
                        self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
                        self.assertTrue(context.check_hostname)
                        self.assertEqual(context.get_ca_certs(), [])

    async def test_invalid_explicit_ca_file_fails_instead_of_falling_back(self):
        with tempfile.TemporaryDirectory() as directory:
            os.environ["SSL_CERT_FILE"] = str(Path(directory) / "missing.pem")
            for factory in (openai.DefaultAsyncHttpxClient, anthropic.DefaultAsyncHttpxClient):
                with self.subTest(sdk=factory.__module__):
                    with self.assertRaises(FileNotFoundError):
                        factory(verify=tls_verification())

    async def test_all_provider_clients_keep_verification_and_sdk_defaults(self):
        translator = Translator.__new__(Translator)
        translator._cfg = Config()
        for provider in ("anthropic", "openai", "google", "deepseek", "ollama"):
            with self.subTest(provider=provider):
                client = translator._build_client(provider)
                try:
                    http_client = client._client
                    context = http_client._transport._pool._ssl_context
                    self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
                    self.assertTrue(context.check_hostname)
                    self.assertTrue(http_client.trust_env)
                    self.assertTrue(http_client.follow_redirects)
                    if provider == "deepseek":
                        self.assertEqual(client.timeout, 45.0)
                        self.assertEqual(client.max_retries, 1)
                finally:
                    await client.close()
                self.assertTrue(http_client.is_closed)

    async def test_environment_proxy_and_no_proxy_still_route_requests(self):
        proxy_requests = []
        direct_requests = []

        def responder(requests):
            async def respond(reader, writer):
                try:
                    request = await reader.readuntil(b"\r\n\r\n")
                    requests.append(request.split(b"\r\n", 1)[0])
                    writer.write(b"HTTP/1.1 204 No Content\r\nContent-Length: 0\r\nConnection: close\r\n\r\n")
                    await writer.drain()
                finally:
                    writer.close()
                    await writer.wait_closed()
            return respond

        proxy = await asyncio.start_server(responder(proxy_requests), "127.0.0.1", 0)
        direct = await asyncio.start_server(responder(direct_requests), "127.0.0.1", 0)
        async with proxy, direct:
            proxy_port = proxy.sockets[0].getsockname()[1]
            direct_port = direct.sockets[0].getsockname()[1]
            os.environ["HTTP_PROXY"] = f"http://127.0.0.1:{proxy_port}"
            os.environ["NO_PROXY"] = "127.0.0.1"
            for factory in (openai.DefaultAsyncHttpxClient, anthropic.DefaultAsyncHttpxClient):
                with self.subTest(sdk=factory.__module__):
                    async with factory(verify=tls_verification(), timeout=5.0) as client:
                        response = await client.get("http://proxy-target.invalid/proxied")
                        self.assertEqual(response.status_code, 204)
                        response = await client.get(f"http://127.0.0.1:{direct_port}/direct")
                        self.assertEqual(response.status_code, 204)
            self.assertEqual(proxy_requests, [b"GET http://proxy-target.invalid/proxied HTTP/1.1"] * 2)
            self.assertEqual(direct_requests, [b"GET /direct HTTP/1.1"] * 2)

    async def test_other_platforms_keep_httpx_default_verification(self):
        with patch("tls_client.sys.platform", "linux"):
            self.assertIs(tls_verification(), True)


if __name__ == "__main__":
    unittest.main()
