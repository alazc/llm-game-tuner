"""Cross-run registry — one row per driver invocation, written to logs/runs.jsonl.

Companion to optimizer/timing.py (which is per-phase WITHIN a single overnight).
This module sits one layer up: every script invocation gets a row in
`logs/runs.jsonl`, so the registry can answer:

  - what runs have I done?
  - how long did each take?
  - which experiment was it part of (MIRROR / TUNER / hazards / ...)?
  - did it finish cleanly?
  - where are the produced artifacts?

Usage in a driver:

    from optimizer import run_log

    def main():
        args = ap.parse_args()
        out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        with run_log.track(experiment='MIRROR-H', script=__file__,
                           args=vars(args), out_dir=out_dir) as run:
            ...
            run.note(f'iter {k}/{K} done')
            run.set(final_score=score)

  - On normal completion: emits `finished` event with status=ok and duration.
  - On exception: emits `failed` event with traceback line; re-raises.
  - On Ctrl+C: emits `failed` event with status=interrupted; re-raises.

Disable for ad-hoc / smoke runs by setting env var RUN_LOG_DISABLE=1 or
passing run_log.disable() before track().
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
import traceback
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


# Repo root inferred from this file's location: optimizer/run_log.py -> ../
REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_LOG = REPO_ROOT / 'logs' / 'runs.jsonl'
DEFAULT_HEARTBEAT_DIR = REPO_ROOT / 'logs' / 'heartbeats'


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds').replace('+00:00', 'Z')


def _utc_now_compact() -> str:
    return datetime.now(timezone.utc).strftime('%Y-%m-%dT%H-%M-%SZ')


def _short_arg_hash(args: Dict[str, Any]) -> str:
    """6-char hash of the arg dict for run_id uniqueness within the same second."""
    blob = json.dumps(args, sort_keys=True, default=str).encode('utf-8')
    return hashlib.sha256(blob).hexdigest()[:6]


def _slug(text: str) -> str:
    out = []
    for ch in text:
        if ch.isalnum() or ch in ('-', '_'):
            out.append(ch)
        else:
            out.append('_')
    return ''.join(out)[:48]


def _atomic_append(path: Path, line: str) -> None:
    """Append one line + newline to path. Single write() call: POSIX (and
    Windows _O_APPEND) guarantee atomicity for writes under PIPE_BUF (4 KB)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = line if line.endswith('\n') else (line + '\n')
    # 'a' opens with O_APPEND; the kernel adjusts the offset atomically.
    with open(path, 'a', encoding='utf-8') as fh:
        fh.write(payload)


def is_enabled() -> bool:
    return os.environ.get('RUN_LOG_DISABLE', '').strip() not in ('1', 'true', 'yes')


def disable() -> None:
    os.environ['RUN_LOG_DISABLE'] = '1'


def enable() -> None:
    os.environ.pop('RUN_LOG_DISABLE', None)


class RunHandle:
    """Mutable run handle returned by track()."""

    def __init__(self, run_id: str, log_path: Path, heartbeat_path: Path,
                 enabled: bool) -> None:
        self.run_id = run_id
        self.log_path = log_path
        self.heartbeat_path = heartbeat_path
        self.enabled = enabled
        # Fields merged into the eventual finished/failed event.
        self._extra: Dict[str, Any] = {}
        self._notes_count = 0
        self._latest_note: Optional[str] = None
        self._t0 = time.perf_counter()

    # --- public API -------------------------------------------------------

    def set(self, **fields: Any) -> None:
        """Merge fields onto the eventual `finished` / `failed` event."""
        self._extra.update(fields)

    def note(self, text: str, **fields: Any) -> None:
        """Emit a `note` event AND touch the heartbeat file. Call from inside
        each iteration so the dashboard can show "iter 5/8" mid-run."""
        self._latest_note = text
        self._notes_count += 1
        if not self.enabled:
            return
        rec = {
            'event': 'note',
            'run_id': self.run_id,
            'ts': _utc_now_iso(),
            'note': text,
            'note_idx': self._notes_count,
        }
        rec.update(fields)
        _atomic_append(self.log_path, json.dumps(rec, default=str))
        # Heartbeat: mtime tells the dashboard "still alive".
        self._touch_heartbeat()

    def _touch_heartbeat(self) -> None:
        if not self.enabled:
            return
        self.heartbeat_path.parent.mkdir(parents=True, exist_ok=True)
        self.heartbeat_path.touch(exist_ok=True)
        # Update mtime to now.
        now = time.time()
        try:
            os.utime(self.heartbeat_path, (now, now))
        except OSError:
            pass

    @property
    def latest_note(self) -> Optional[str]:
        return self._latest_note


@contextmanager
def track(experiment: str,
          script: str,
          args: Optional[Dict[str, Any]] = None,
          out_dir: Optional[Path] = None,
          *,
          condition: Optional[str] = None,
          backend: Optional[str] = None,
          log_path: Optional[Path] = None,
          extra: Optional[Dict[str, Any]] = None,
          ) -> Iterator[RunHandle]:
    """Context manager: writes started/finished/failed events to logs/runs.jsonl.

    Parameters
    ----------
    experiment: str
        Free-form tag — 'MIRROR-H', 'TUNER', 'ARCHITECT', 'hazards', 'novelty',
        'phase-a', 'phase-a-robustness', 'T-SANITY', 'GA-2p', etc. Required.
    script: str
        Driver path. `__file__` is the right value at the call site.
    args: dict, optional
        Echoed CLI args. `vars(argparse.Namespace)` is the typical input.
        Non-JSON-serialisable values are coerced via str().
    out_dir: Path, optional
        Where this run's artifacts land. Stored so the dashboard can render
        a clickable link.
    condition: str, optional
        TUNER ablation cell (T-CANON / T-MUTE / T-HAZ / ... ) or any other
        sub-categorisation. Set automatically from `args['ablation_condition']`
        if present and `condition` is None.
    backend: str, optional
        'local' / 'heuristic' / 'anthropic' / 'openai' — drawn from args['backend']
        if present and `backend` is None. Useful for filtering smokes out.
    log_path: Path, optional
        Override `logs/runs.jsonl` location (used by tests).
    extra: dict, optional
        Any other static fields to attach to the started event.

    Yields
    ------
    RunHandle
        Handle with `.run_id`, `.note(text)`, `.set(**fields)`.
    """
    enabled = is_enabled()
    log_path = Path(log_path) if log_path is not None else DEFAULT_LOG
    args_dict: Dict[str, Any] = dict(args or {})
    # Auto-derive condition / backend from args if not explicitly passed.
    if condition is None:
        condition = args_dict.get('ablation_condition') or args_dict.get('condition')
    if backend is None:
        backend = args_dict.get('backend')

    run_id = '{ts}_{exp}_{cond}_{h}'.format(
        ts=_utc_now_compact(),
        exp=_slug(experiment),
        cond=_slug(condition) if condition else 'na',
        h=_short_arg_hash(args_dict),
    )
    heartbeat_dir = DEFAULT_HEARTBEAT_DIR
    if log_path != DEFAULT_LOG:
        heartbeat_dir = log_path.parent / 'heartbeats'
    heartbeat_path = heartbeat_dir / run_id

    started_payload: Dict[str, Any] = {
        'event': 'started',
        'run_id': run_id,
        'ts': _utc_now_iso(),
        'experiment': experiment,
        'script': str(script),
        'condition': condition,
        'backend': backend,
        'out_dir': str(out_dir) if out_dir is not None else None,
        'args': args_dict,
        'pid': os.getpid(),
        'host': os.environ.get('COMPUTERNAME') or os.environ.get('HOSTNAME') or '',
        'cmdline': ' '.join(sys.argv),
    }
    if extra:
        started_payload.update(extra)

    if enabled:
        try:
            _atomic_append(log_path, json.dumps(started_payload, default=str))
        except OSError as e:
            # Never let logging failure crash a run. Print and continue.
            print(f'[run_log] WARN: could not write started event: {e}',
                  file=sys.stderr, flush=True)

    handle = RunHandle(run_id=run_id, log_path=log_path,
                       heartbeat_path=heartbeat_path, enabled=enabled)
    handle._touch_heartbeat()

    t0 = time.perf_counter()
    try:
        yield handle
    except KeyboardInterrupt:
        _emit_terminal(handle, status='interrupted',
                       elapsed=time.perf_counter() - t0,
                       error_kind='KeyboardInterrupt',
                       error_msg='user interrupted (Ctrl+C)')
        raise
    except SystemExit as e:
        # SystemExit with non-zero code is a failure; zero is a clean exit.
        code = e.code if isinstance(e.code, int) else (0 if e.code is None else 1)
        if code == 0:
            _emit_terminal(handle, status='ok',
                           elapsed=time.perf_counter() - t0)
        else:
            _emit_terminal(handle, status='failed',
                           elapsed=time.perf_counter() - t0,
                           error_kind='SystemExit',
                           error_msg=f'exit code {code}')
        raise
    except BaseException as e:
        tb_last = traceback.format_exc().strip().splitlines()[-1] if traceback.format_exc().strip() else str(e)
        _emit_terminal(handle, status='failed',
                       elapsed=time.perf_counter() - t0,
                       error_kind=type(e).__name__,
                       error_msg=tb_last[:500])
        raise
    else:
        _emit_terminal(handle, status='ok',
                       elapsed=time.perf_counter() - t0)


def _emit_terminal(handle: RunHandle, status: str, elapsed: float,
                   error_kind: Optional[str] = None,
                   error_msg: Optional[str] = None) -> None:
    if not handle.enabled:
        return
    rec: Dict[str, Any] = {
        'event': 'failed' if status in ('failed', 'interrupted') else 'finished',
        'run_id': handle.run_id,
        'ts': _utc_now_iso(),
        'status': status,
        'duration_s': round(elapsed, 3),
        'note_count': handle._notes_count,
        'latest_note': handle._latest_note,
    }
    if error_kind is not None:
        rec['error_kind'] = error_kind
    if error_msg is not None:
        rec['error_msg'] = error_msg
    rec.update(handle._extra)
    try:
        _atomic_append(handle.log_path, json.dumps(rec, default=str))
    except OSError as e:
        print(f'[run_log] WARN: could not write terminal event: {e}',
              file=sys.stderr, flush=True)
