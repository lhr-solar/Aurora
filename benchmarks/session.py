

"""Benchmarks session utilities.

A *session* is a persistent output folder under `data/benchmarks/` that is reused
across repeated benchmark runs as long as the *suite signature* (scenarios/budgets
and key suite knobs) stays the same.

The runner writes:
- `<out_dir>/current_session.json` (tracks the active session)
- `<out_dir>/latest_session_path.txt` (absolute or relative path to the session dir)
- `<session_dir>/session_meta.json`

The UI can use this module to:
- compute a suite signature
- resolve the active session directory
- create a new session if the signature changes
"""

from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Sequence


def _short_hash(s: str, n: int = 8) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:n]


def compute_suite_signature(payload: Dict[str, Any]) -> str:
    """Return a stable signature string from a JSON-serializable payload.

    The caller (runner or UI) should pass a payload that defines *comparability*
    across runs (e.g., scenarios/budgets/total_time/gmpp_ref settings).

    Notes:
    - Uses `sort_keys=True` for stability.
    - Falls back to `repr()` for non-JSON types.
    """

    def _default(o: Any) -> str:
        return repr(o)

    return json.dumps(payload, sort_keys=True, default=_default)


def make_suite_payload(
    *,
    scenarios: Sequence[Dict[str, Any]],
    budgets: Sequence[Dict[str, Any]],
    total_time: float,
    gmpp_ref: bool,
    gmpp_ref_period_s: float,
    gmpp_ref_points: int,
) -> Dict[str, Any]:
    """Create a canonical suite payload for signature computation.

    This function is intentionally typed to plain dicts so both runner and UI can
    call it without importing runner dataclasses.
    """
    return {
        "scenarios": list(scenarios),
        "budgets": list(budgets),
        "total_time": float(total_time),
        "gmpp_ref": bool(gmpp_ref),
        "gmpp_ref_period_s": float(gmpp_ref_period_s),
        "gmpp_ref_points": int(gmpp_ref_points),
    }


def load_current_session(out_dir: Path) -> Optional[Dict[str, Any]]:
    p = out_dir / "current_session.json"
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def save_current_session(out_dir: Path, info: Dict[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "current_session.json").write_text(
        json.dumps(info, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def read_latest_session_path(out_dir: Path) -> Optional[Path]:
    p = out_dir / "latest_session_path.txt"
    if not p.exists():
        return None
    try:
        txt = p.read_text(encoding="utf-8").strip()
        if not txt:
            return None
        sd = Path(txt)
        # If runner wrote a relative path, resolve relative to out_dir
        if not sd.is_absolute():
            sd = (out_dir / sd).resolve()
        return sd
    except Exception:
        return None


def ensure_session_dir(
    *,
    out_dir: Path,
    signature: str,
    create_if_missing: bool = True,
) -> Optional[Path]:
    """Return active session dir if signature matches, else create a new one.

    If `create_if_missing=False`, returns None when there is no matching session.
    """
    sig_hash = _short_hash(signature)
    cur = load_current_session(out_dir)

    if cur:
        try:
            if str(cur.get("signature_hash")) == sig_hash and str(cur.get("signature")) == signature:
                session_name = str(cur.get("session_name"))
                if session_name:
                    sd = out_dir / session_name
                    if sd.exists():
                        return sd
        except Exception:
            pass

    if not create_if_missing:
        return None

    ts = time.strftime("%Y%m%d_%H%M%S")
    session_name = f"session_{ts}__{sig_hash}"
    session_dir = out_dir / session_name
    session_dir.mkdir(parents=True, exist_ok=True)

    meta = {
        "created_at": ts,
        "session_name": session_name,
        "signature_hash": sig_hash,
        "signature": signature,
    }

    (session_dir / "session_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    save_current_session(out_dir, meta)

    # Keep UI convenience pointer in sync
    (out_dir / "latest_session_path.txt").write_text(str(session_dir), encoding="utf-8")

    return session_dir


def list_session_dirs(out_dir: Path) -> Sequence[Path]:
    """List all session directories under out_dir."""
    if not out_dir.exists():
        return []
    out: list[Path] = []
    for p in out_dir.iterdir():
        if p.is_dir() and p.name.startswith("session_"):
            out.append(p)
    return sorted(out)


def append_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> None:
    """Append rows to a JSONL file (create if missing)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")