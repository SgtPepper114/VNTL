import asyncio
import os
import ssl
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from config import Config
from context_manager import ContextManager
from translator import Translator, _safe_error_summary


def client_with_text(provider, text):
    if provider == "anthropic":
        response = SimpleNamespace(
            content=[SimpleNamespace(type="text", text=text)], stop_reason="end_turn")
    else:
        response = SimpleNamespace(choices=[SimpleNamespace(
            message=SimpleNamespace(content=text), finish_reason="stop")])
    create = AsyncMock(return_value=response)
    return SimpleNamespace(
        messages=SimpleNamespace(create=create),
        chat=SimpleNamespace(completions=SimpleNamespace(create=create)),
        close=AsyncMock(),
    )


class ProxyRefreshTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.proxies = {"https": "http://old-route.invalid:8080", "no": "localhost"}
        self.proxy_patch = patch("translator.urllib.request.getproxies",
                                 side_effect=lambda: dict(self.proxies))
        self.proxy_patch.start()
        self.addCleanup(self.proxy_patch.stop)
        self.env_patch = patch.dict(os.environ)
        self.env_patch.start()
        self.addCleanup(self.env_patch.stop)
        os.environ.pop("SSL_CERT_FILE", None)
        os.environ.pop("SSL_CERT_DIR", None)

    def make_translator(self, provider="deepseek"):
        cfg = Config()
        for role in ("translator", "descriptor", "summarizer"):
            setattr(cfg, f"{role}_provider", provider)
        client = client_with_text(provider, "old response")
        with patch.object(Translator, "_build_client", return_value=client):
            translator = Translator(cfg, ContextManager())
        return translator, client

    async def call(self, translator, provider="deepseek"):
        return await translator._call(provider, "fixture-model", "system", [], 128)

    async def test_unchanged_settings_reuse_client_and_ignore_unrelated_environment(self):
        translator, old = self.make_translator()
        with patch.object(translator, "_build_client") as build:
            for _ in range(3):
                self.assertEqual(await self.call(translator), "old response")
            os.environ["UNRELATED_FIXTURE_SETTING"] = "changed"
            self.proxies["ftp"] = "irrelevant-route.invalid"
            await self.call(translator)
            build.assert_not_called()
        old.close.assert_not_awaited()
        self.assertFalse(translator._active_clients)

    async def test_route_change_recreates_each_supported_provider_once(self):
        for provider in ("anthropic", "openai", "google", "deepseek", "ollama"):
            with self.subTest(provider=provider):
                translator, old = self.make_translator(provider)
                replacement = client_with_text(provider, "new response")
                self.proxies["https"] += "/changed"
                with patch.object(translator, "_build_client", return_value=replacement) as build:
                    self.assertEqual(await self.call(translator, provider), "new response")
                    self.assertEqual(await self.call(translator, provider), "new response")
                    build.assert_called_once_with(provider)
                old.close.assert_awaited_once()
                replacement.close.assert_not_awaited()

    async def test_bypass_and_certificate_environment_changes_refresh_clients(self):
        translator, old = self.make_translator()
        changes = (
            lambda: self.proxies.update(no="localhost,api.deepseek.com"),
            lambda: os.environ.update(SSL_CERT_FILE="PRIVATE_CERT_PATH"),
            lambda: os.environ.update(SSL_CERT_DIR="PRIVATE_CERT_DIR"),
            lambda: os.environ.pop("SSL_CERT_FILE"),
        )
        for change in changes:
            replacement = client_with_text("deepseek", "new response")
            change()
            with patch.object(translator, "_build_client", return_value=replacement) as build:
                await self.call(translator)
                build.assert_called_once_with("deepseek")
            old.close.assert_awaited_once()
            replacement.close.assert_not_awaited()
            old = replacement

    async def test_two_inflight_calls_keep_old_transport_until_both_complete(self):
        translator, old = self.make_translator()
        entered = [asyncio.Event(), asyncio.Event()]
        release = [asyncio.Event(), asyncio.Event()]
        count = 0
        response = old.chat.completions.create.return_value

        async def pending(**kwargs):
            nonlocal count
            index = count
            count += 1
            entered[index].set()
            await release[index].wait()
            return response

        old.chat.completions.create.side_effect = pending
        first = asyncio.create_task(self.call(translator))
        await entered[0].wait()
        second = asyncio.create_task(self.call(translator))
        await entered[1].wait()
        replacement = client_with_text("deepseek", "new response")
        self.proxies["https"] = "http://new-route.invalid:8080"
        with patch.object(translator, "_build_client", return_value=replacement):
            self.assertEqual(await self.call(translator), "new response")
        old.close.assert_not_awaited()
        release[0].set()
        self.assertEqual(await first, "old response")
        old.close.assert_not_awaited()
        release[1].set()
        self.assertEqual(await second, "old response")
        old.close.assert_awaited_once()
        replacement.close.assert_not_awaited()
        self.assertFalse(translator._retired_clients)
        self.assertFalse(translator._active_clients)

    async def test_config_rebuild_and_cancellation_release_correct_client(self):
        translator, old = self.make_translator()
        entered = asyncio.Event()

        async def pending(**kwargs):
            entered.set()
            await asyncio.Event().wait()

        old.chat.completions.create.side_effect = pending
        task = asyncio.create_task(self.call(translator))
        await entered.wait()
        replacement = client_with_text("deepseek", "new response")
        with patch.object(translator, "_build_client", return_value=replacement):
            translator.rebuild_clients()
        old.close.assert_not_awaited()
        self.assertEqual(await self.call(translator), "new response")
        task.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await task
        old.close.assert_awaited_once()
        replacement.close.assert_not_awaited()
        self.assertFalse(translator._active_clients)

    async def test_partial_rebuild_failure_keeps_generation_and_retries_next_request(self):
        translator, old = self.make_translator()
        translator._clients["google"] = old
        replacement = client_with_text("deepseek", "new response")
        signature = translator._network_signature
        self.proxies["https"] = "http://new-route.invalid:8080"
        with patch.object(translator, "_build_client",
                          side_effect=[replacement, RuntimeError("PRIVATE_SETUP_SECRET")]):
            with self.assertLogs("translator", level="WARNING") as logs:
                with self.assertRaises(RuntimeError):
                    await self.call(translator)
        self.assertNotIn("PRIVATE_", "\n".join(logs.output))
        self.assertEqual(translator._network_signature, signature)
        self.assertIs(translator._clients["deepseek"], old)
        old.close.assert_not_awaited()
        replacement.close.assert_awaited_once()
        recovered = client_with_text("deepseek", "recovered")
        with patch.object(translator, "_build_client", return_value=recovered):
            self.assertEqual(await self.call(translator), "recovered")
        old.close.assert_awaited_once()

    async def test_error_diagnostics_include_tls_cause_without_secrets(self):
        translator, old = self.make_translator()
        tls = ssl.SSLCertVerificationError(1, "PRIVATE_TLS_MESSAGE")
        tls.verify_code = 19
        tls.verify_message = "PRIVATE_CERTIFICATE_NAME"
        connection = ConnectionError("https://PRIVATE_HOST/?key=PRIVATE_KEY")
        connection.__cause__ = tls
        old.chat.completions.create.side_effect = connection
        with self.assertLogs("translator", level="WARNING") as logs:
            with self.assertRaises(ConnectionError):
                await self.call(translator)
        diagnostic = "\n".join(logs.output)
        self.assertIn("ConnectionError -> SSLCertVerificationError", diagnostic)
        self.assertIn("self-signed certificate in chain", diagnostic)
        self.assertNotIn("PRIVATE_", diagnostic)
        self.assertNotIn("https://", diagnostic)
        self.assertFalse(translator._active_clients)

    async def test_refresh_and_cleanup_logs_never_include_proxy_credentials(self):
        translator, old = self.make_translator()
        old.close.side_effect = RuntimeError("PRIVATE_CLOSE_SECRET")
        self.proxies["https"] = "http://PRIVATE_USER:PRIVATE_PASSWORD@PRIVATE_HOST:8080"
        os.environ["SSL_CERT_FILE"] = "PRIVATE_CA_PATH"
        replacement = client_with_text("deepseek", "new response")
        with patch.object(translator, "_build_client", return_value=replacement):
            with self.assertLogs("translator", level="INFO") as logs:
                self.assertEqual(await self.call(translator), "new response")
        diagnostic = "\n".join(logs.output)
        self.assertIn("refreshed API clients", diagnostic)
        self.assertIn("cleanup failed: RuntimeError", diagnostic)
        self.assertNotIn("PRIVATE_", diagnostic)
        self.assertNotIn("http://", diagnostic)

    async def test_client_builders_preserve_default_tls_verification_and_environment(self):
        translator, _ = self.make_translator()
        with patch("anthropic.AsyncAnthropic") as anthropic_client, \
                patch("openai.AsyncOpenAI") as openai_client:
            for provider in ("anthropic", "openai", "google", "deepseek", "ollama"):
                translator._build_client(provider)
            calls = anthropic_client.call_args_list + openai_client.call_args_list
        self.assertEqual(len(calls), 5)
        for call in calls:
            transport = call.kwargs["http_client"]
            self.assertTrue(transport.trust_env)
            context = transport._transport._pool._ssl_context
            self.assertTrue(context.check_hostname)
            self.assertEqual(context.verify_mode, ssl.CERT_REQUIRED)
            await transport.aclose()

    def test_error_summary_bounds_cyclic_chains_and_redacts_unknown_tls_reason(self):
        error = ssl.SSLCertVerificationError(1, "PRIVATE_MESSAGE")
        error.verify_code = 99999
        error.verify_message = "PRIVATE_CUSTOM_REASON"
        error.__cause__ = error
        self.assertEqual(_safe_error_summary(error),
                         "SSLCertVerificationError (certificate verification failed)")


if __name__ == "__main__":
    unittest.main()
