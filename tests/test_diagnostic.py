"""Tests du diagnostic matériel, sans Ollama et sans accès réseau."""
from __future__ import annotations

import pathlib
import sys
import unittest
from unittest import mock

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

import diagnostic  # noqa: E402


class NvidiaParsingTests(unittest.TestCase):
    def test_parse_multiple_gpus_and_memory(self) -> None:
        output = (
            "NVIDIA GeForce RTX 4060, 8192, 7168\n"
            "NVIDIA GeForce RTX 3060, 12288, 11000\n"
        )
        gpus = diagnostic.parse_nvidia_smi_output(output)
        self.assertEqual(len(gpus), 2)
        self.assertEqual(gpus[0]["name"], "NVIDIA GeForce RTX 4060")
        self.assertEqual(gpus[0]["total_bytes"], 8192 * diagnostic.MIB)
        self.assertEqual(gpus[0]["free_bytes"], 7168 * diagnostic.MIB)
        self.assertEqual(gpus[1]["total_bytes"], 12288 * diagnostic.MIB)

    def test_parse_unknown_vram_without_crashing(self) -> None:
        gpus = diagnostic.parse_nvidia_smi_output("NVIDIA Shared GPU, N/A, N/A\n")
        self.assertEqual(len(gpus), 1)
        self.assertIsNone(gpus[0]["total_bytes"])
        self.assertIsNone(gpus[0]["free_bytes"])

    def test_parse_rocm_smi(self) -> None:
        output = (
            "GPU[0] : VRAM Total Memory (B): 8589934592\n"
            "GPU[0] : VRAM Total Used Memory (B): 2147483648\n"
        )
        gpu = diagnostic.parse_rocm_smi_output(output)[0]
        self.assertEqual(gpu["total_bytes"], 8 * diagnostic.GIB)
        self.assertEqual(gpu["free_bytes"], 6 * diagnostic.GIB)


class MemoryEstimateTests(unittest.TestCase):
    def setUp(self) -> None:
        self.model_info = {
            "general.architecture": "qwen3",
            "qwen3.block_count": 36,
            "qwen3.embedding_length": 4096,
            "qwen3.attention.head_count": 32,
            "qwen3.attention.head_count_kv": 8,
        }
        self.gpu = {
            "name": "Test GPU",
            "total_bytes": 8 * diagnostic.GIB,
            "free_bytes": 8 * diagnostic.GIB,
        }

    def test_q8_kv_cache_estimate_uses_layers_and_kv_heads(self) -> None:
        expected = int(36 * 8192 * 8 * (4096 / 32) * 2 * (34 / 32))
        self.assertEqual(diagnostic.estimate_kv_cache_bytes(self.model_info), expected)

    def test_context_scales_kv_cache_estimate(self) -> None:
        small = diagnostic.estimate_kv_cache_bytes(self.model_info, context=4096)
        large = diagnostic.estimate_kv_cache_bytes(self.model_info, context=8192)
        self.assertEqual(large, 2 * small)

    def test_missing_metadata_returns_unknown(self) -> None:
        self.assertIsNone(diagnostic.estimate_kv_cache_bytes({}, context=8192, kv_cache="q8_0"))

    def test_vram_budget_keeps_a_reserve(self) -> None:
        result = diagnostic.compute_vram_budget(self.gpu)
        self.assertEqual(result["reserve_bytes"], int(0.8 * diagnostic.GIB))
        self.assertEqual(result["usable_bytes"], 8 * diagnostic.GIB - int(0.8 * diagnostic.GIB))

    def test_model_fit_is_estimate_not_a_boolean_when_data_missing(self) -> None:
        result = diagnostic.estimate_model_fit(4 * diagnostic.GIB, {}, self.gpu)
        self.assertIsNone(result["fits_estimate"])

    def test_model_fit_accounts_for_weights_kv_and_runtime(self) -> None:
        fits = diagnostic.estimate_model_fit(
            5 * diagnostic.GIB, self.model_info, self.gpu, context=8192, kv_cache="q8_0"
        )
        too_large = diagnostic.estimate_model_fit(
            8 * diagnostic.GIB, self.model_info, self.gpu, context=8192, kv_cache="q8_0"
        )
        self.assertTrue(fits["fits_estimate"])
        self.assertFalse(too_large["fits_estimate"])


class ToolCallingScoringTests(unittest.TestCase):
    def test_expected_call_with_exact_arguments_passes(self) -> None:
        case = diagnostic.BENCH_CASES[0]
        response = {
            "eval_count": 12,
            "eval_duration": 2_000_000_000,
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup_lab_asset",
                            "arguments": {"asset_id": "LAB-ASSET-001"},
                        }
                    }
                ]
            },
        }
        result = diagnostic.score_tool_case(case, response)
        self.assertTrue(result["passed"])
        self.assertFalse(result["tools_executed"])
        self.assertEqual(result["tokens_per_second"], 6.0)

    def test_expected_call_rejects_extra_arguments(self) -> None:
        case = diagnostic.BENCH_CASES[0]
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup_lab_asset",
                            "arguments": {"asset_id": "LAB-ASSET-001", "target": "extra"},
                        }
                    }
                ]
            }
        }
        self.assertFalse(diagnostic.score_tool_case(case, response)["passed"])

    def test_json_encoded_arguments_are_parsed(self) -> None:
        case = diagnostic.BENCH_CASES[0]
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup_lab_asset",
                            "arguments": '{"asset_id":"LAB-ASSET-001"}',
                        }
                    }
                ]
            }
        }
        self.assertTrue(diagnostic.score_tool_case(case, response)["passed"])

    def test_no_call_trap_passes_only_when_no_tool_is_called(self) -> None:
        case = diagnostic.BENCH_CASES[2]
        self.assertTrue(diagnostic.score_tool_case(case, {"message": {"content": "443"}})["passed"])
        self.assertFalse(diagnostic.score_tool_case(case, {"message": {"content": ""}})["passed"])
        response = {
            "message": {
                "tool_calls": [
                    {
                        "function": {
                            "name": "lookup_lab_asset",
                            "arguments": {"asset_id": "LAB-ASSET-003"},
                        }
                    }
                ]
            }
        }
        self.assertFalse(diagnostic.score_tool_case(case, response)["passed"])

    def test_multiple_calls_fail_when_exactly_one_is_expected(self) -> None:
        case = diagnostic.BENCH_CASES[0]
        response = {
            "message": {
                "tool_calls": [
                    {"function": {"name": "lookup_lab_asset", "arguments": {"asset_id": "LAB-ASSET-001"}}},
                    {"function": {"name": "lookup_lab_asset", "arguments": {"asset_id": "LAB-ASSET-001"}}},
                ]
            }
        }
        self.assertFalse(diagnostic.score_tool_case(case, response)["passed"])


class LocalApiBoundaryTests(unittest.TestCase):
    def test_ollama_connection_is_fixed_to_loopback(self) -> None:
        connections: list[tuple[str, int]] = []

        class FakeResponse:
            status = 200

            @staticmethod
            def read() -> bytes:
                return b'{"models": []}'

        class FakeConnection:
            def __init__(self, host: str, port: int, timeout: float) -> None:
                connections.append((host, port))

            def request(self, method: str, path: str, body: bytes | None = None, headers=None) -> None:
                self.requested_path = path

            @staticmethod
            def getresponse() -> FakeResponse:
                return FakeResponse()

            @staticmethod
            def close() -> None:
                return None

        with mock.patch.object(diagnostic.http.client, "HTTPConnection", FakeConnection):
            response = diagnostic._ollama_request("GET", "/api/tags")
        self.assertEqual(response, {"models": []})
        self.assertEqual(connections, [("127.0.0.1", 11434)])

    def test_api_path_rejects_external_url(self) -> None:
        with self.assertRaises(diagnostic.DiagnosticError):
            diagnostic._ollama_request("GET", "http://example.com/api/tags")

    def test_benchmark_uses_only_synthetic_calls_and_confirms_unload(self) -> None:
        chat_responses = [
            {"load_duration": 123, "message": {"tool_calls": [{"function": {"name": "lookup_lab_asset", "arguments": {"asset_id": "LAB-ASSET-001"}}}]}},
            {"message": {"tool_calls": [{"function": {"name": "lookup_service_record", "arguments": {"asset_id": "LAB-ASSET-002", "service": "http"}}}]}},
            {"message": {"content": "443", "tool_calls": []}},
        ]
        chat_index = 0
        calls: list[tuple[str, str, dict | None]] = []

        def fake_request(method: str, path: str, payload=None, timeout=8.0):
            nonlocal chat_index
            calls.append((method, path, payload))
            if method == "GET" and path == "/api/tags":
                return {"models": [{"name": "test-model:latest", "size": 2 * diagnostic.GIB, "details": {}}]}
            if method == "POST" and path == "/api/show":
                return {"model_info": {}}
            if method == "GET" and path == "/api/ps":
                return {"models": []}
            if method == "POST" and path == "/api/chat":
                response = chat_responses[chat_index]
                chat_index += 1
                return response
            if method == "POST" and path == "/api/generate":
                if payload and payload.get("keep_alive") == 0:
                    return {"response": ""}
                return {"eval_count": 100, "eval_duration": 10_000_000_000, "load_duration": 123}
            raise AssertionError(f"Appel inattendu : {method} {path}")

        with (
            mock.patch.object(diagnostic, "collect_hardware", return_value={"gpus": [], "warnings": []}),
            mock.patch.object(diagnostic, "_ollama_request", side_effect=fake_request),
        ):
            report = diagnostic.run_model_benchmark("test-model", 8192, "q8_0", 1, 1)

        self.assertTrue(report["tool_calling_passed"])
        self.assertEqual(len(report["tool_calling_cases"]), 3)
        self.assertTrue(all(not item["tools_executed"] for item in report["tool_calling_cases"]))
        self.assertTrue(report["model_unload_request_sent_by_script"])
        self.assertTrue(report["model_unload_confirmed"])
        self.assertEqual(chat_index, 3)
        self.assertTrue(all(path in {"/api/tags", "/api/show", "/api/ps", "/api/chat", "/api/generate"} for _, path, _ in calls))


if __name__ == "__main__":
    unittest.main()
