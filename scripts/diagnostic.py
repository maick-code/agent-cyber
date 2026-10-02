#!/usr/bin/env python3
"""Diagnostic matériel et modèles Ollama, sans dépendance tierce.

Le script n'utilise que des informations locales : commandes de diagnostic
matériel en lecture seule et API Ollama sur 127.0.0.1:11434. Le benchmark de
tool calling envoie des outils fictifs au modèle, mais ne les exécute jamais.
Aucune option ne télécharge de modèle ni ne contacte une API distante.
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import datetime as _datetime
import http.client
import io
import json
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from typing import Any

MIB = 1024**2
GIB = 1024**3
OLLAMA_HOST = "127.0.0.1"
OLLAMA_PORT = 11434
OLLAMA_BASE_URL = f"http://{OLLAMA_HOST}:{OLLAMA_PORT}"
DEFAULT_TIMEOUT = 8.0
DEFAULT_BENCH_TIMEOUT = 300.0
RUNTIME_OVERHEAD_BYTES = 384 * MIB
MIN_VRAM_RESERVE_BYTES = 512 * MIB


class DiagnosticError(RuntimeError):
    """Erreur récupérable affichable sans traceback à l'utilisateur."""


def _run_command(args: list[str], timeout: float = 3.0) -> str | None:
    """Exécute une commande locale de diagnostic, sans shell ni effet d'écriture."""
    try:
        result = subprocess.run(
            args,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


def _parse_mib(value: str) -> int | None:
    value = value.strip().replace(",", "")
    if not value or value.lower() in {"n/a", "na", "[not supported]"}:
        return None
    try:
        return int(float(value) * MIB)
    except ValueError:
        return None


def parse_nvidia_smi_output(output: str) -> list[dict[str, Any]]:
    """Parse name,total MiB,free MiB from nvidia-smi's CSV output."""
    gpus: list[dict[str, Any]] = []
    for row in csv.reader(io.StringIO(output)):
        if not row:
            continue
        name = row[0].strip()
        if not name or name.lower() == "name":
            continue
        total = _parse_mib(row[1]) if len(row) > 1 else None
        free = _parse_mib(row[2]) if len(row) > 2 else None
        gpus.append(
            {
                "name": name,
                "vendor": "NVIDIA",
                "total_bytes": total,
                "free_bytes": free,
                "memory_type": "dedicated",
                "source": "nvidia-smi",
            }
        )
    return gpus


def parse_rocm_smi_output(output: str) -> list[dict[str, Any]]:
    """Parse the common `rocm-smi --showmeminfo vram` output format."""
    totals: dict[str, int] = {}
    used: dict[str, int] = {}
    for line in output.splitlines():
        gpu_match = re.search(r"GPU\[(\d+)\]", line, re.IGNORECASE)
        if not gpu_match:
            continue
        gpu_id = gpu_match.group(1)
        value_match = re.search(r":\s*([\d,]+)\s*$", line)
        if not value_match:
            continue
        value = int(value_match.group(1).replace(",", ""))
        if re.search(r"VRAM Total Memory", line, re.IGNORECASE):
            totals[gpu_id] = value
        elif re.search(r"VRAM Total Used Memory", line, re.IGNORECASE):
            used[gpu_id] = value

    result: list[dict[str, Any]] = []
    for gpu_id in sorted(totals, key=lambda item: int(item)):
        total = totals[gpu_id]
        used_bytes = used.get(gpu_id)
        result.append(
            {
                "name": f"AMD GPU {gpu_id}",
                "vendor": "AMD",
                "total_bytes": total,
                "free_bytes": max(0, total - used_bytes) if used_bytes is not None else None,
                "memory_type": "dedicated-or-driver-reported",
                "source": "rocm-smi",
            }
        )
    return result


def _linux_gpus_from_sysfs() -> list[dict[str, Any]]:
    """Read Linux DRM device IDs and AMD VRAM counters where available."""
    root = "/sys/class/drm"
    if not os.path.isdir(root):
        return []
    result: list[dict[str, Any]] = []
    try:
        card_paths = sorted(
            os.path.join(root, name)
            for name in os.listdir(root)
            if re.fullmatch(r"card\d+", name)
        )
    except OSError:
        return []

    vendor_names = {"0x1002": "AMD", "0x8086": "Intel", "0x10de": "NVIDIA"}
    for card in card_paths:
        device_dir = os.path.join(card, "device")
        try:
            with open(os.path.join(device_dir, "vendor"), encoding="ascii") as f:
                vendor_id = f.read().strip().lower()
            with open(os.path.join(device_dir, "device"), encoding="ascii") as f:
                device_id = f.read().strip().lower()
        except OSError:
            continue
        vendor = vendor_names.get(vendor_id)
        if vendor is None or vendor == "NVIDIA":
            continue  # NVIDIA est normalement mesuré plus précisément par nvidia-smi.

        total_path = os.path.join(device_dir, "mem_info_vram_total")
        used_path = os.path.join(device_dir, "mem_info_vram_used")
        total = used = None
        try:
            with open(total_path, encoding="ascii") as f:
                total = int(f.read().strip())
            with open(used_path, encoding="ascii") as f:
                used = int(f.read().strip())
        except (OSError, ValueError):
            pass

        has_dedicated_vram = total is not None and total > 0
        result.append(
            {
                "name": f"{vendor} GPU ({device_id})",
                "vendor": vendor,
                "total_bytes": total if has_dedicated_vram else None,
                "free_bytes": max(0, total - used) if has_dedicated_vram and used is not None else None,
                "memory_type": "dedicated" if has_dedicated_vram else "shared-or-unknown",
                "source": "Linux DRM sysfs",
            }
        )
    return result


def _windows_gpus_from_powershell() -> list[dict[str, Any]]:
    """Read display adapter names; WMI's 32-bit VRAM value is not trusted."""
    if platform.system() != "Windows" or not shutil.which("powershell"):
        return []
    command = (
        "Get-CimInstance Win32_VideoController | "
        "Select-Object -Property Name | ConvertTo-Json -Compress"
    )
    output = _run_command(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", command], timeout=5.0
    )
    if not output:
        return []
    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return []
    if isinstance(data, dict):
        data = [data]
    if not isinstance(data, list):
        return []
    return [
        {
            "name": str(item.get("Name") or "Windows display adapter"),
            "vendor": "unknown",
            "total_bytes": None,
            "free_bytes": None,
            "memory_type": "unknown (WMI VRAM is not reliable for modern GPUs)",
            "source": "Windows CIM; name only",
        }
        for item in data
        if isinstance(item, dict)
    ]


def _parse_system_profiler_size(value: Any) -> int | None:
    if not isinstance(value, str):
        return None
    match = re.search(r"([\d.]+)\s*(KB|MB|GB|TB)\b", value, re.IGNORECASE)
    if not match:
        return None
    amount = float(match.group(1))
    unit = match.group(2).upper()
    multiplier = {"KB": 1024, "MB": MIB, "GB": GIB, "TB": 1024**4}[unit]
    return int(amount * multiplier)


def _mac_gpus() -> list[dict[str, Any]]:
    if platform.system() != "Darwin" or not shutil.which("system_profiler"):
        return []
    output = _run_command(["system_profiler", "SPDisplaysDataType", "-json"], timeout=8.0)
    if not output:
        return []
    try:
        root = json.loads(output)
    except json.JSONDecodeError:
        return []
    displays = root.get("SPDisplaysDataType", []) if isinstance(root, dict) else []
    result: list[dict[str, Any]] = []
    for item in displays:
        if not isinstance(item, dict):
            continue
        name = item.get("sppci_model") or item.get("_name") or "Apple display adapter"
        vram = _parse_system_profiler_size(item.get("spdisplays_vram"))
        shared = "spdisplays_vram_shared" in item or "Apple" in str(name)
        result.append(
            {
                "name": str(name),
                "vendor": "Apple" if "Apple" in str(name) else "unknown",
                "total_bytes": vram if not shared else None,
                "free_bytes": None,
                "memory_type": "unified/shared; distinct VRAM not reported" if shared else "dedicated-or-driver-reported",
                "source": "system_profiler",
            }
        )
    return result


def detect_gpus() -> list[dict[str, Any]]:
    """Detect GPU names and VRAM without installing or downloading anything."""
    gpus: list[dict[str, Any]] = []

    if shutil.which("nvidia-smi"):
        output = _run_command(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free",
                "--format=csv,noheader,nounits",
            ],
            timeout=4.0,
        )
        if output:
            gpus.extend(parse_nvidia_smi_output(output))

    if not gpus and shutil.which("rocm-smi"):
        output = _run_command(["rocm-smi", "--showmeminfo", "vram"], timeout=5.0)
        if output:
            gpus.extend(parse_rocm_smi_output(output))

    if not gpus:
        gpus.extend(_linux_gpus_from_sysfs())
    if not gpus:
        gpus.extend(_mac_gpus())
    if not gpus:
        gpus.extend(_windows_gpus_from_powershell())
    return gpus


def _linux_ram() -> dict[str, Any] | None:
    path = "/proc/meminfo"
    if not os.path.isfile(path):
        return None
    values: dict[str, int] = {}
    try:
        with open(path, encoding="ascii") as f:
            for line in f:
                match = re.match(r"^(MemTotal|MemAvailable|MemFree|Buffers|Cached):\s+(\d+)\s+kB", line)
                if match:
                    values[match.group(1)] = int(match.group(2)) * 1024
    except OSError:
        return None
    total = values.get("MemTotal")
    if total is None:
        return None
    available = values.get("MemAvailable")
    if available is None:
        available = sum(values.get(key, 0) for key in ("MemFree", "Buffers", "Cached")) or None
    return {"total_bytes": total, "available_bytes": available, "source": "/proc/meminfo"}


def _windows_ram() -> dict[str, Any] | None:
    if platform.system() != "Windows":
        return None

    class MEMORYSTATUSEX(ctypes.Structure):
        _fields_ = [
            ("dwLength", ctypes.c_ulong),
            ("dwMemoryLoad", ctypes.c_ulong),
            ("ullTotalPhys", ctypes.c_ulonglong),
            ("ullAvailPhys", ctypes.c_ulonglong),
            ("ullTotalPageFile", ctypes.c_ulonglong),
            ("ullAvailPageFile", ctypes.c_ulonglong),
            ("ullTotalVirtual", ctypes.c_ulonglong),
            ("ullAvailVirtual", ctypes.c_ulonglong),
            ("ullAvailExtendedVirtual", ctypes.c_ulonglong),
        ]

    try:
        status = MEMORYSTATUSEX()
        status.dwLength = ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            return None
        return {
            "total_bytes": int(status.ullTotalPhys),
            "available_bytes": int(status.ullAvailPhys),
            "source": "GlobalMemoryStatusEx",
        }
    except (AttributeError, OSError):
        return None


def _mac_ram() -> dict[str, Any] | None:
    if platform.system() != "Darwin":
        return None
    total = None
    try:
        total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        pass
    if total is None:
        output = _run_command(["sysctl", "-n", "hw.memsize"], timeout=2.0)
        if output and output.isdigit():
            total = int(output)
    available = None
    source = "sysconf/sysctl"
    vm_stat = _run_command(["vm_stat"], timeout=2.0)
    if vm_stat:
        page_match = re.search(r"page size of (\d+) bytes", vm_stat)
        page_size = int(page_match.group(1)) if page_match else 4096
        page_counts = 0
        for label in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
            match = re.search(rf"^{re.escape(label)}:\s*(\d+)\.", vm_stat, re.MULTILINE)
            if match:
                page_counts += int(match.group(1))
        if page_counts:
            available = page_counts * page_size
            source = "sysconf/sysctl + vm_stat (available memory approximative)"
    if total is None:
        return None
    return {"total_bytes": total, "available_bytes": available, "source": source}


def _portable_sysconf_ram() -> dict[str, Any] | None:
    try:
        total = int(os.sysconf("SC_PHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        return None
    try:
        available = int(os.sysconf("SC_AVPHYS_PAGES")) * int(os.sysconf("SC_PAGE_SIZE"))
    except (ValueError, OSError, AttributeError):
        available = None
    return {"total_bytes": total, "available_bytes": available, "source": "os.sysconf"}


def detect_ram() -> dict[str, Any] | None:
    if platform.system() == "Linux":
        return _linux_ram() or _portable_sysconf_ram()
    if platform.system() == "Windows":
        return _windows_ram() or _portable_sysconf_ram()
    if platform.system() == "Darwin":
        return _mac_ram() or _portable_sysconf_ram()
    return _portable_sysconf_ram()


def compute_vram_budget(gpu: dict[str, Any]) -> dict[str, Any]:
    """Compute a conservative budget after reserving driver/desktop headroom."""
    total = gpu.get("total_bytes")
    if not isinstance(total, int) or total <= 0:
        return {
            "gpu": gpu.get("name"),
            "total_bytes": total,
            "free_bytes": gpu.get("free_bytes"),
            "reserve_bytes": None,
            "usable_bytes": None,
            "note": "VRAM dédiée non mesurée ; impossible de calculer un budget fiable.",
        }
    free = gpu.get("free_bytes")
    free_known = isinstance(free, int) and free >= 0
    available = min(free, total) if free_known else total
    reserve = max(MIN_VRAM_RESERVE_BYTES, int(total * 0.10))
    return {
        "gpu": gpu.get("name"),
        "total_bytes": total,
        "free_bytes": free if free_known else None,
        "free_assumed_bytes": None if free_known else total,
        "reserve_bytes": reserve,
        "usable_bytes": max(0, available - reserve),
        "note": "Budget prudent ; réserve max(512 MiB, 10 % VRAM).",
    }


def _find_model_value(model_info: dict[str, Any], suffix: str) -> int | float | None:
    architecture = model_info.get("general.architecture")
    keys = []
    if architecture:
        keys.append(f"{architecture}.{suffix}")
    keys.extend(key for key in model_info if key.endswith("." + suffix) and key not in keys)
    for key in keys:
        value = model_info.get(key)
        if isinstance(value, (int, float)) and value > 0:
            return value
    return None


KV_BYTES_PER_VALUE = {
    "q8_0": 34 / 32,  # 32 quantized values + 2-byte scale
    "q4_0": 18 / 32,
    "f16": 2.0,
    "bf16": 2.0,
    "f32": 4.0,
}


def estimate_kv_cache_bytes(
    model_info: dict[str, Any], context: int = 8192, kv_cache: str = "q8_0"
) -> int | None:
    """Estimate KV cache from layer/head metadata; None when metadata is missing."""
    if context <= 0 or kv_cache not in KV_BYTES_PER_VALUE:
        return None
    layers = _find_model_value(model_info, "block_count")
    embedding = _find_model_value(model_info, "embedding_length")
    heads = _find_model_value(model_info, "attention.head_count")
    kv_heads = _find_model_value(model_info, "attention.head_count_kv") or heads
    if not all(isinstance(value, (int, float)) and value > 0 for value in (layers, embedding, heads, kv_heads)):
        return None
    head_dim = embedding / heads
    if head_dim <= 0:
        return None
    # K et V ; l'estimation q8_0 inclut l'overhead de mise à l'échelle par bloc.
    size = layers * context * kv_heads * head_dim * 2 * KV_BYTES_PER_VALUE[kv_cache]
    return int(size)


def estimate_model_fit(
    model_size_bytes: int | None,
    model_info: dict[str, Any] | None,
    gpu: dict[str, Any],
    context: int = 8192,
    kv_cache: str = "q8_0",
) -> dict[str, Any]:
    budget = compute_vram_budget(gpu)
    kv_bytes = estimate_kv_cache_bytes(model_info or {}, context, kv_cache)
    if model_size_bytes is None or budget["usable_bytes"] is None or kv_bytes is None:
        return {
            "fits_estimate": None,
            "model_file_bytes": model_size_bytes,
            "kv_cache_estimate_bytes": kv_bytes,
            "runtime_overhead_bytes": RUNTIME_OVERHEAD_BYTES,
            "required_estimate_bytes": None,
            "usable_vram_bytes": budget["usable_bytes"],
            "context": context,
            "kv_cache_type": kv_cache,
            "note": "Estimation incomplète : taille du modèle, VRAM ou métadonnées KV manquantes.",
        }
    required = model_size_bytes + kv_bytes + RUNTIME_OVERHEAD_BYTES
    return {
        "fits_estimate": required <= budget["usable_bytes"],
        "model_file_bytes": model_size_bytes,
        "kv_cache_estimate_bytes": kv_bytes,
        "runtime_overhead_bytes": RUNTIME_OVERHEAD_BYTES,
        "required_estimate_bytes": required,
        "usable_vram_bytes": budget["usable_bytes"],
        "context": context,
        "kv_cache_type": kv_cache,
        "note": "Heuristique : taille du GGUF + KV + 384 MiB ; la répartition CPU/GPU et le pilote peuvent changer le résultat.",
    }


def _format_bytes(value: int | None) -> str:
    if not isinstance(value, int):
        return "inconnu"
    if value >= GIB:
        return f"{value / GIB:.2f} Gio"
    if value >= MIB:
        return f"{value / MIB:.0f} Mio"
    return f"{value} octets"


def _human_hardware_report(report: dict[str, Any]) -> str:
    lines = ["Diagnostic matériel local", "=" * 26]
    lines.append(f"Système : {report['platform']['system']} {report['platform']['release']} ({report['platform']['machine']})")
    ram = report.get("ram")
    if ram:
        lines.append(
            f"RAM : {_format_bytes(ram.get('total_bytes'))} total ; "
            f"{_format_bytes(ram.get('available_bytes'))} disponible ({ram.get('source')})"
        )
    else:
        lines.append("RAM : mesure indisponible")
    gpus = report.get("gpus", [])
    if not gpus:
        lines.append("GPU : aucune information exploitable détectée")
    for gpu in gpus:
        lines.append(
            f"GPU : {gpu.get('name')} — VRAM {_format_bytes(gpu.get('total_bytes'))}, "
            f"libre {_format_bytes(gpu.get('free_bytes'))} ({gpu.get('source')})"
        )
        budget = compute_vram_budget(gpu)
        lines.append(f"  Budget prudent estimé pour modèle + cache : {_format_bytes(budget.get('usable_bytes'))}")
    lines.append("Ollama : non interrogé (utiliser --liste ou --bench pour l'API locale).")
    if report.get("warnings"):
        lines.append("Avertissements :")
        lines.extend(f"  - {warning}" for warning in report["warnings"])
    return "\n".join(lines)


def collect_hardware() -> dict[str, Any]:
    gpus = detect_gpus()
    ram = detect_ram()
    warnings: list[str] = []
    if not gpus:
        warnings.append("GPU non détecté via les interfaces prises en charge ; confirmer le modèle manuellement.")
    elif not any(isinstance(gpu.get("total_bytes"), int) for gpu in gpus):
        warnings.append("GPU détecté, mais VRAM dédiée indisponible ou mémoire unifiée ; estimation par modèle impossible.")
    if ram is None:
        warnings.append("La mémoire vive n'a pas pu être mesurée par les interfaces standard du système.")
    return {
        "captured_at": _datetime.datetime.now(_datetime.timezone.utc).isoformat(),
        "platform": {
            "system": platform.system(),
            "release": platform.release(),
            "machine": platform.machine(),
            "python": platform.python_version(),
        },
        "ram": ram,
        "gpus": gpus,
        "vram_budgets": [compute_vram_budget(gpu) for gpu in gpus],
        "warnings": warnings,
        "local_only": True,
        "ollama_contacted": False,
    }


def _ollama_request(
    method: str, path: str, payload: dict[str, Any] | None = None, timeout: float = DEFAULT_TIMEOUT
) -> dict[str, Any]:
    """Talk directly to Ollama on loopback; never follows redirects or uses proxies."""
    if not path.startswith("/") or ".." in path or "://" in path:
        raise DiagnosticError("Chemin d'API Ollama invalide.")
    connection = http.client.HTTPConnection(OLLAMA_HOST, OLLAMA_PORT, timeout=timeout)
    body = None if payload is None else json.dumps(payload, ensure_ascii=False).encode("utf-8")
    headers = {"Accept": "application/json"}
    if body is not None:
        headers["Content-Type"] = "application/json; charset=utf-8"
    try:
        connection.request(method, path, body=body, headers=headers)
        response = connection.getresponse()
        response_body = response.read()
    except (OSError, http.client.HTTPException) as exc:
        raise DiagnosticError(
            f"Ollama local est injoignable sur {OLLAMA_BASE_URL} ({exc.__class__.__name__}). "
            "Vérifie que le daemon Ollama local est démarré."
        ) from None
    finally:
        connection.close()

    try:
        data = json.loads(response_body.decode("utf-8")) if response_body else {}
    except (UnicodeDecodeError, json.JSONDecodeError):
        data = {"_raw": response_body.decode("utf-8", errors="replace")[:500]}
    if response.status < 200 or response.status >= 300:
        detail = data.get("error") if isinstance(data, dict) else None
        raise DiagnosticError(f"Ollama {method} {path} a répondu HTTP {response.status}: {detail or 'erreur API'}")
    if not isinstance(data, dict):
        raise DiagnosticError(f"Réponse Ollama inattendue pour {path} (objet JSON attendu).")
    return data


def _normal_model_name(name: str) -> str:
    return name[:-7] if name.endswith(":latest") else name


def _find_tag(tags: list[dict[str, Any]], model: str) -> dict[str, Any] | None:
    requested = _normal_model_name(model)
    for item in tags:
        for key in ("name", "model"):
            value = item.get(key)
            if isinstance(value, str) and _normal_model_name(value) == requested:
                return item
    return None


def _model_info(model: str, timeout: float) -> dict[str, Any]:
    data = _ollama_request("POST", "/api/show", {"model": model, "verbose": True}, timeout)
    info = data.get("model_info", {})
    return info if isinstance(info, dict) else {}


def _list_loaded_models(timeout: float) -> list[dict[str, Any]]:
    data = _ollama_request("GET", "/api/ps", timeout=timeout)
    items = data.get("models", [])
    if not isinstance(items, list):
        raise DiagnosticError("Réponse /api/ps invalide : liste de modèles attendue.")
    result: list[dict[str, Any]] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        result.append(
            {
                "name": item.get("name") or item.get("model"),
                "size_bytes": item.get("size"),
                "size_vram_bytes": item.get("size_vram"),
                "expires_at": item.get("expires_at"),
            }
        )
    return result


def _fit_for_gpus(
    size_bytes: int | None,
    info: dict[str, Any],
    gpus: list[dict[str, Any]],
    context: int,
    kv_cache: str,
) -> list[dict[str, Any]]:
    results = []
    for gpu in gpus:
        results.append(
            {
                "gpu": gpu.get("name"),
                **estimate_model_fit(size_bytes, info, gpu, context, kv_cache),
            }
        )
    return results


def list_local_models(context: int, kv_cache: str, timeout: float) -> dict[str, Any]:
    hardware = collect_hardware()
    tags_data = _ollama_request("GET", "/api/tags", timeout=timeout)
    tags = tags_data.get("models", [])
    if not isinstance(tags, list):
        raise DiagnosticError("Réponse /api/tags invalide : liste de modèles attendue.")
    models = []
    for item in tags:
        if not isinstance(item, dict):
            continue
        name = item.get("name") or item.get("model") or "(nom inconnu)"
        size = item.get("size") if isinstance(item.get("size"), int) else None
        warnings: list[str] = []
        try:
            info = _model_info(str(name), timeout)
        except DiagnosticError as exc:
            info = {}
            warnings.append(str(exc))
        models.append(
            {
                "name": name,
                "size_bytes": size,
                "modified_at": item.get("modified_at"),
                "details": item.get("details", {}),
                "fit_estimates": _fit_for_gpus(size, info, hardware["gpus"], context, kv_cache),
                "warnings": warnings,
            }
        )
    return {
        "captured_at": hardware["captured_at"],
        "ollama_base_url": OLLAMA_BASE_URL,
        "models": models,
        "hardware": hardware,
        "estimation_config": {"context": context, "kv_cache_type": kv_cache},
        "note": "Taille du modèle depuis Ollama ; l'estimation VRAM inclut KV et marge, mais reste approximative.",
        "local_only": True,
    }


def _asset_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "lookup_lab_asset",
            "description": "Outil fictif du benchmark ; retourne une fiche d'actif de laboratoire synthétique.",
            "parameters": {
                "type": "object",
                "properties": {"asset_id": {"type": "string"}},
                "required": ["asset_id"],
                "additionalProperties": False,
            },
        },
    }


def _service_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "lookup_service_record",
            "description": "Outil fictif du benchmark ; retourne une fiche de service synthétique.",
            "parameters": {
                "type": "object",
                "properties": {
                    "asset_id": {"type": "string"},
                    "service": {"type": "string", "enum": ["http", "https"]},
                },
                "required": ["asset_id", "service"],
                "additionalProperties": False,
            },
        },
    }


BENCH_CASES: tuple[dict[str, Any], ...] = (
    {
        "id": "appel_attendu_actif_synthetique",
        "prompt": "Dans ce test fictif, récupère la fiche de l'actif LAB-ASSET-001 avec l'outil lookup_lab_asset. Appelle cet outil exactement une fois et n'invente pas le résultat.",
        "tools": [_asset_tool()],
        "expected_tool": "lookup_lab_asset",
        "expected_args": {"asset_id": "LAB-ASSET-001"},
    },
    {
        "id": "appel_attendu_service_synthetique",
        "prompt": "Dans ce test fictif, récupère la fiche HTTP de l'actif LAB-ASSET-002 au moyen de lookup_service_record. Appelle cet outil exactement une fois avec les arguments correspondant à cet actif et au service http.",
        "tools": [_service_tool()],
        "expected_tool": "lookup_service_record",
        "expected_args": {"asset_id": "LAB-ASSET-002", "service": "http"},
    },
    {
        "id": "piege_sur_appel_aucun_outil",
        "prompt": "N'appelle aucun outil. Réponds uniquement à partir des données déjà fournies : actif fictif LAB-ASSET-003, port TCP 443, service https. Quel est le port TCP ? Réponds avec le nombre seulement.",
        "tools": [_asset_tool(), _service_tool()],
        "expected_tool": None,
        "expected_args": None,
        "expected_text_contains": "443",
    },
)


BENCH_SYSTEM_PROMPT = (
    "Tu participes à un test synthétique de sélection d'outils. Les identifiants LAB-ASSET-* "
    "sont fictifs. Les outils décrits ne seront jamais exécutés : on mesure seulement si tu "
    "émets un appel structuré approprié. N'invente jamais le résultat d'un outil."
)


def _normalize_tool_calls(message: dict[str, Any]) -> tuple[list[dict[str, Any]], str | None]:
    raw_calls = message.get("tool_calls", [])
    if raw_calls is None:
        return [], None
    if not isinstance(raw_calls, list):
        return [], "tool_calls n'est pas une liste"
    calls: list[dict[str, Any]] = []
    for raw in raw_calls:
        if not isinstance(raw, dict):
            calls.append({"name": None, "arguments": None})
            continue
        function = raw.get("function", {})
        if not isinstance(function, dict):
            calls.append({"name": None, "arguments": None})
            continue
        args = function.get("arguments", {})
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except json.JSONDecodeError:
                args = None
        calls.append({"name": function.get("name"), "arguments": args})
    return calls, None


def score_tool_case(case: dict[str, Any], response: dict[str, Any]) -> dict[str, Any]:
    message = response.get("message", {})
    if not isinstance(message, dict):
        message = {}
    calls, parse_error = _normalize_tool_calls(message)
    expected_tool = case["expected_tool"]
    failure: str | None = parse_error
    if failure is None and expected_tool is None:
        if calls:
            failure = f"sur-appel : {len(calls)} appel(s) d'outil observé(s), aucun attendu"
        else:
            expected_text = case.get("expected_text_contains")
            content = message.get("content", "")
            if expected_text and (not isinstance(content, str) or expected_text not in content):
                failure = f"aucun appel d'outil (correct), mais la réponse n'inclut pas {expected_text!r}"
    elif failure is None:
        if len(calls) != 1:
            failure = f"{len(calls)} appel(s) observé(s), exactement un attendu"
        elif calls[0].get("name") != expected_tool:
            failure = f"outil observé {calls[0].get('name')!r}, attendu {expected_tool!r}"
        elif not isinstance(calls[0].get("arguments"), dict):
            failure = "arguments d'outil absents ou non structurés"
        elif calls[0]["arguments"] != case["expected_args"]:
            failure = f"arguments observés {calls[0]['arguments']!r}, attendus {case['expected_args']!r}"
    return {
        "case_id": case["id"],
        "passed": failure is None,
        "expected_tool": expected_tool,
        "expected_args": case["expected_args"],
        "observed_tool_calls": calls,
        "error": failure,
        "eval_count": response.get("eval_count"),
        "eval_duration_ns": response.get("eval_duration"),
        "tokens_per_second": _tokens_per_second(response.get("eval_count"), response.get("eval_duration")),
        "tools_executed": False,
    }


def _tokens_per_second(eval_count: Any, eval_duration_ns: Any) -> float | None:
    if not isinstance(eval_count, (int, float)) or not isinstance(eval_duration_ns, (int, float)):
        return None
    if eval_count < 0 or eval_duration_ns <= 0:
        return None
    return round(eval_count / (eval_duration_ns / 1_000_000_000), 2)


def _same_model(left: Any, right: str) -> bool:
    return isinstance(left, str) and _normal_model_name(left) == _normal_model_name(right)


def run_model_benchmark(
    model: str,
    context: int,
    kv_cache: str,
    timeout: float,
    bench_timeout: float = DEFAULT_BENCH_TIMEOUT,
) -> dict[str, Any]:
    hardware = collect_hardware()
    tags_data = _ollama_request("GET", "/api/tags", timeout=timeout)
    tags = tags_data.get("models", [])
    if not isinstance(tags, list):
        raise DiagnosticError("Réponse /api/tags invalide.")
    tag = _find_tag([item for item in tags if isinstance(item, dict)], model)
    if tag is None:
        raise DiagnosticError(
            f"Le modèle {model!r} n'est pas installé localement dans Ollama. "
            "Le diagnostic ne le téléchargera pas ; consulte --liste."
        )
    actual_name = str(tag.get("name") or tag.get("model") or model)
    model_size = tag.get("size") if isinstance(tag.get("size"), int) else None
    info = _model_info(actual_name, timeout)
    loaded = _list_loaded_models(timeout)
    conflicting = [entry.get("name") for entry in loaded if not _same_model(entry.get("name"), actual_name)]
    if conflicting:
        raise DiagnosticError(
            "Un autre modèle Ollama est déjà résident ("
            + ", ".join(str(name) for name in conflicting)
            + "). Pour respecter un seul modèle chargé, arrête-le toi-même puis relance le diagnostic."
        )
    was_loaded = any(_same_model(entry.get("name"), actual_name) for entry in loaded)

    results: list[dict[str, Any]] = []
    warnings: list[str] = []
    load_duration_ns: int | None = None
    throughput: dict[str, Any] = {}
    unload_error: str | None = None
    unload_request_sent = False
    unload_confirmed: bool | None = None
    try:
        for index, case in enumerate(BENCH_CASES):
            request = {
                "model": actual_name,
                "stream": False,
                "keep_alive": "5m",
                "messages": [
                    {"role": "system", "content": BENCH_SYSTEM_PROMPT},
                    {"role": "user", "content": case["prompt"]},
                ],
                "tools": case["tools"],
                "options": {"temperature": 0, "num_ctx": context, "num_predict": 160},
            }
            response = _ollama_request("POST", "/api/chat", request, timeout=bench_timeout)
            if index == 0 and isinstance(response.get("load_duration"), int):
                load_duration_ns = response["load_duration"]
            results.append(score_tool_case(case, response))

        throughput_prompt = (
            "Explique en français, de façon neutre et sans code, la différence entre la mémoire vive "
            "et la mémoire vidéo. Produis environ cent mots."
        )
        start = time.perf_counter()
        generated = _ollama_request(
            "POST",
            "/api/generate",
            {
                "model": actual_name,
                "prompt": throughput_prompt,
                "stream": False,
                "keep_alive": "5m",
                "options": {"temperature": 0, "num_ctx": context, "num_predict": 128},
            },
            timeout=bench_timeout,
        )
        wall_seconds = time.perf_counter() - start
        throughput = {
            "eval_count": generated.get("eval_count"),
            "eval_duration_ns": generated.get("eval_duration"),
            "tokens_per_second": _tokens_per_second(generated.get("eval_count"), generated.get("eval_duration")),
            "wall_seconds_including_api_overhead": round(wall_seconds, 3),
            "load_duration_ns": generated.get("load_duration"),
        }
        if throughput["tokens_per_second"] is None:
            warnings.append("Ollama n'a pas fourni eval_count/eval_duration ; tok/s non calculable.")
    finally:
        # Ne décharge que le modèle chargé par ce diagnostic ; ne touche pas à celui déjà résident.
        if not was_loaded:
            try:
                _ollama_request(
                    "POST",
                    "/api/generate",
                    {"model": actual_name, "prompt": "", "keep_alive": 0, "stream": False},
                    timeout=timeout,
                )
                unload_request_sent = True
                remaining = _list_loaded_models(timeout)
                unload_confirmed = not any(
                    _same_model(entry.get("name"), actual_name) for entry in remaining
                )
                if not unload_confirmed:
                    warnings.append("Ollama a accepté la demande de déchargement, mais le modèle apparaît encore dans /api/ps.")
            except DiagnosticError as exc:
                unload_error = str(exc)
                warnings.append("Le déchargement demandé n'a pas pu être confirmé ; vérifie `ollama ps` manuellement.")

    fit_estimates = _fit_for_gpus(model_size, info, hardware["gpus"], context, kv_cache)
    passed = all(result["passed"] for result in results)
    return {
        "model": actual_name,
        "model_size_bytes": model_size,
        "model_details": tag.get("details", {}),
        "context": context,
        "kv_cache_type_for_estimate": kv_cache,
        "kv_cache_setting_note": "q8_0/q4_0 est ici une hypothèse d'estimation ; le script ne change pas la configuration du daemon Ollama.",
        "fit_estimates": fit_estimates,
        "model_loaded_before_test": was_loaded,
        "loaded_models_before_test": loaded,
        "ollama_runtime_settings": {
            "OLLAMA_FLASH_ATTENTION": "1 (attendu, non vérifiable via l'API Ollama)",
            "OLLAMA_MAX_LOADED_MODELS": "1 (attendu ; le script vérifie seulement la liste /api/ps avant le test)",
        },
        "tool_calling_cases": results,
        "tool_calling_passed": passed,
        "throughput": throughput,
        "first_request_load_duration_ns": load_duration_ns,
        "tools_executed": False,
        "model_unload_request_sent_by_script": unload_request_sent,
        "model_unload_confirmed": unload_confirmed,
        "unload_error": unload_error,
        "warnings": warnings,
        "hardware": hardware,
        "local_only": True,
    }


def _human_models_report(report: dict[str, Any]) -> str:
    lines = ["Modèles installés dans Ollama local", "=" * 35]
    models = report.get("models", [])
    if not models:
        lines.append("Aucun modèle local trouvé.")
    for item in models:
        lines.append(f"- {item['name']} — {_format_bytes(item.get('size_bytes'))} sur disque")
        estimates = item.get("fit_estimates", [])
        if not estimates:
            lines.append("  Estimation VRAM indisponible (GPU ou métadonnées manquants).")
        for fit in estimates:
            status = "probable" if fit.get("fits_estimate") is True else "insuffisant" if fit.get("fits_estimate") is False else "indéterminé"
            lines.append(
                f"  {fit.get('gpu')}: {status} ; besoin estimé "
                f"{_format_bytes(fit.get('required_estimate_bytes'))}, budget prudent "
                f"{_format_bytes(fit.get('usable_vram_bytes'))}"
            )
        for warning in item.get("warnings", []):
            lines.append(f"  Avertissement : {warning}")
    lines.append(
        f"Estimation basée sur contexte {report['estimation_config']['context']} et cache "
        f"{report['estimation_config']['kv_cache_type']} ; ce n'est pas une garantie de chargement."
    )
    return "\n".join(lines)


def _human_benchmark_report(report: dict[str, Any]) -> str:
    lines = [f"Benchmark Ollama local — {report['model']}", "=" * (23 + len(report["model"]))]
    for result in report["tool_calling_cases"]:
        status = "OK" if result["passed"] else "ÉCHEC"
        lines.append(f"{status} — {result['case_id']}")
        if result.get("error"):
            lines.append(f"  {result['error']}")
        if result.get("tokens_per_second") is not None:
            lines.append(f"  génération : {result['tokens_per_second']} tokens/s")
    throughput = report.get("throughput", {})
    lines.append(
        "Débit texte neutre : "
        + (f"{throughput['tokens_per_second']} tokens/s" if throughput.get("tokens_per_second") is not None else "indisponible")
    )
    lines.append(f"Durée de chargement initiale : {_format_duration(report.get('first_request_load_duration_ns'))}")
    if report.get("fit_estimates"):
        for fit in report["fit_estimates"]:
            state = "probable" if fit.get("fits_estimate") is True else "insuffisant" if fit.get("fits_estimate") is False else "indéterminé"
            lines.append(f"VRAM {fit.get('gpu')} : {state} ({fit.get('note')})")
    lines.append("Aucun outil déclaré au modèle n'a été exécuté.")
    if report.get("warnings"):
        lines.extend(f"Avertissement : {warning}" for warning in report["warnings"])
    return "\n".join(lines)


def _format_duration(value_ns: Any) -> str:
    if not isinstance(value_ns, (int, float)) or value_ns < 0:
        return "inconnue"
    return f"{value_ns / 1_000_000_000:.2f} s"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Diagnostic matériel et Ollama local (bibliothèque standard uniquement).",
        epilog=(
            "Exemples : python scripts/diagnostic.py --json ; "
            "python scripts/diagnostic.py --liste ; "
            "python scripts/diagnostic.py --bench qwen3:8b"
        ),
    )
    actions = parser.add_mutually_exclusive_group()
    actions.add_argument("--liste", action="store_true", help="lister les modèles déjà installés dans Ollama local")
    actions.add_argument("--bench", metavar="MODELE", help="tester un modèle Ollama déjà installé (aucun outil n'est exécuté)")
    parser.add_argument("--json", action="store_true", help="sortie JSON, sans texte décoratif")
    parser.add_argument("--ctx", type=int, default=8192, help="contexte simulé pour le benchmark et l'estimation (défaut : 8192)")
    parser.add_argument(
        "--kv-cache",
        choices=tuple(KV_BYTES_PER_VALUE),
        default="q8_0",
        help="type de cache KV supposé pour l'estimation uniquement (défaut : q8_0)",
    )
    parser.add_argument("--timeout", type=float, default=DEFAULT_TIMEOUT, help="timeout des appels de métadonnées locales, en secondes")
    parser.add_argument(
        "--bench-timeout",
        type=float,
        default=DEFAULT_BENCH_TIMEOUT,
        help="timeout par génération pendant --bench, en secondes (défaut : 300)",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.ctx <= 0:
        parser.error("--ctx doit être un entier positif")
    if args.timeout <= 0:
        parser.error("--timeout doit être strictement positif")
    if args.bench_timeout <= 0:
        parser.error("--bench-timeout doit être strictement positif")

    try:
        if args.liste:
            report = list_local_models(args.ctx, args.kv_cache, args.timeout)
            print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else _human_models_report(report))
            return 0
        if args.bench:
            report = run_model_benchmark(
                args.bench, args.ctx, args.kv_cache, args.timeout, args.bench_timeout
            )
            print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else _human_benchmark_report(report))
            return 0 if report["tool_calling_passed"] else 1
        report = collect_hardware()
        print(json.dumps(report, ensure_ascii=False, indent=2) if args.json else _human_hardware_report(report))
        return 0
    except DiagnosticError as exc:
        error_report = {"ok": False, "error": str(exc), "local_only": True}
        if args.json:
            print(json.dumps(error_report, ensure_ascii=False, indent=2))
        else:
            print(f"Erreur : {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
