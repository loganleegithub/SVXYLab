"""有界的官方文件下载与本地缓存；失败响应也保留，不生成替代行情。"""

from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit


def public_url(url: str) -> str:
    parts = urlsplit(url)
    query = urlencode([(k, "REDACTED" if k.lower() in {"apikey", "crumb", "token", "access_token"} else v)
                       for k, v in parse_qsl(parts.query, keep_blank_values=True)])
    return urlunsplit((parts.scheme, parts.netloc, parts.path, query, ""))


class DownloadStore:
    def __init__(self, root: Path):
        self.root = root
        self.directory = root / "data" / "raw"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.manifest = self.directory / "downloads.jsonl"
        self.records = [json.loads(line) for line in self.manifest.read_text().splitlines()] if self.manifest.exists() else []

    def save_response(self, body: bytes, *, url: str, source: str, label: str,
                      http_status: int, client: str, error_layer: str = "") -> dict:
        """保存成熟 Python 客户端实际收到的正文，不保存 cookie 或请求认证头。"""
        now = datetime.now(timezone.utc)
        digest = sha256(body).hexdigest()
        destination = self.directory / source / f"{label}-{now.strftime('%Y%m%dT%H%M%S%fZ')}-{digest[:12]}.response"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(body)
        destination.chmod(0o444)
        record = {"url": public_url(url), "effective_url": public_url(url), "source": source, "label": label,
                  "retrieved_at": now.isoformat(), "method": client, "attempt": 1, "http_status": http_status,
                  "curl_exit_code": None, "error_layer": error_layer or ("" if 200 <= http_status < 300 else "HTTP"),
                  "ok": 200 <= http_status < 300 and not error_layer, "bytes": len(body), "sha256": digest,
                  "raw_file": destination.relative_to(self.root).as_posix()}
        with self.manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        print(f"{source}/{label}: HTTP {http_status}, {len(body)} bytes, {record['error_layer'] or 'OK'}", flush=True)
        return record

    def import_local(self, path: Path, *, url: str, source: str, label: str,
                     expected_sha256: str | None = None) -> dict:
        content = path.read_bytes()
        digest = sha256(content).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(f"本机原件摘要与已有清单不符：{path.name}")
        for record in reversed(self.records):
            if record["url"] == public_url(url) and record["sha256"] == digest:
                return record
        now = datetime.now(timezone.utc)
        destination = self.directory / source / f"{label}-{digest[:12]}{path.suffix}"
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(content)
        destination.chmod(0o444)
        record = {"url": public_url(url), "effective_url": public_url(url), "source": source,
                  "label": label, "retrieved_at": now.isoformat(), "imported_at": now.isoformat(),
                  "original_downloaded_at": None, "imported_from": str(path), "attempt": 0,
                  "http_status": None, "curl_exit_code": None, "error_layer": "", "ok": True,
                  "bytes": len(content), "sha256": digest, "raw_file": destination.relative_to(self.root).as_posix(),
                  "expected_sha256": expected_sha256, "method": "LOCAL_IMPORT"}
        with self.manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(record, ensure_ascii=False) + "\n")
        self.records.append(record)
        return record

    def fetch(self, url: str, *, source: str, label: str, refresh: bool = False,
              http1: bool = False, attempts: int = 2) -> dict:
        if attempts not in (1, 2):
            raise ValueError("每个请求只允许一至两次尝试")
        safe_url = public_url(url)
        if not refresh:
            for record in reversed(self.records):
                if record["url"] == safe_url and record["ok"]:
                    path = self.root / record["raw_file"]
                    if path.is_file() and sha256(path.read_bytes()).hexdigest() == record["sha256"]:
                        return record
        scratch = self.root / ".cache" / "downloads"
        scratch.mkdir(parents=True, exist_ok=True)
        for attempt in range(1, attempts + 1):
            started = datetime.now(timezone.utc)
            stamp = started.strftime("%Y%m%dT%H%M%S%fZ")
            temporary = scratch / f"{stamp}.body"
            # URL 经 stdin 传入；密钥不出现在进程参数、清单或 stdout。
            args = ["/usr/bin/curl", "--disable", "--config", "-", "--location", "--max-redirs", "3",
                    "--proto", "=https", "--proto-redir", "=https", "--connect-timeout", "8",
                    "--max-time", "30", "--retry", "0", "--silent", "--show-error",
                    "--output", str(temporary), "--write-out", "%{json}"]
            if http1:
                args.append("--http1.1")
            try:
                result = subprocess.run(args, input="url = " + json.dumps(url) + "\n", text=True,
                                        capture_output=True, timeout=35)
                info = json.loads(result.stdout) if result.stdout else {}
                code = result.returncode
            except subprocess.TimeoutExpired:
                info, code = {}, 124
            content = temporary.read_bytes() if temporary.exists() else b""
            digest = sha256(content).hexdigest()
            suffix = Path(urlsplit(url).path).suffix.lower()
            if suffix not in {".csv", ".json", ".pdf", ".zip"}:
                suffix = ".html" if "html" in (info.get("content_type") or "") else ".response"
            destination = self.directory / source / f"{label}-{stamp}-{digest[:12]}{suffix}"
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(content)
            destination.chmod(0o444)
            if temporary.exists():
                temporary.unlink()
            http = info.get("http_code", 0)
            error = {5: "PROXY_DNS", 6: "DNS", 7: "CONNECTION", 28: "TIMEOUT", 35: "TLS",
                     60: "TLS_CERTIFICATE", 92: "HTTP2_STREAM", 124: "PROCESS_TIMEOUT"}.get(code, "CURL" if code else "")
            if not code and not 200 <= http < 300:
                error = "HTTP"
            record = {"url": safe_url, "effective_url": public_url(info.get("url_effective", safe_url)),
                      "source": source, "label": label, "retrieved_at": datetime.now(timezone.utc).isoformat(),
                      "request_started_at": started.isoformat(), "attempt": attempt, "http_status": http,
                      "curl_exit_code": code, "error_layer": error, "ok": code == 0 and 200 <= http < 300,
                      "content_type": info.get("content_type"), "bytes": len(content), "sha256": digest,
                      "raw_file": destination.relative_to(self.root).as_posix(),
                      "elapsed_seconds": info.get("time_total")}
            record["http_version_requested"] = "1.1" if http1 else "curl_default"
            with self.manifest.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            self.records.append(record)
            print(f"{source}/{label}: HTTP {http}, {len(content)} bytes, {error or 'OK'}", flush=True)
            if record["ok"] or (code == 0 and http not in {429, 500, 502, 503, 504}) or attempt == attempts:
                return record
        raise RuntimeError("不可达分支")
