"""Wall-clock instrumentation for the round-1 overnight runs.

Round 1 builds compute claims into the report ("MIRROR-H ran in 8 hr; LLM
generation dominates eval cost by 2 orders of magnitude"). Those claims need
measurements, not hand-waved estimates. `WallClockLogger` is the single
sink: every phase, every iteration, every LLM call, every bootstrap CI.

Usage:
    logger = WallClockLogger(out_dir='report/figures/round1')
    with logger.time('phase', event='phase_start', phase='MIRROR-H'):
        ...
    logger.record('phase', event='llm_call', gen_tokens=42, gen_seconds=0.31)
    logger.update_status({'MIRROR-H': 'complete', 'TUNER': 'in_progress', ...})

The JSONL file is the source of truth; STATUS.md is a human-readable
dashboard regenerated each phase change.
"""
from __future__ import annotations

import json
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterator, Optional


class WallClockLogger:
    """Append-only JSONL logger + STATUS.md emitter.

    Granularities:
      - phase     : phase_start / phase_end events
      - iteration : iter_start / iter_end events keyed by (phase, board, seed, iter)
      - sub-step  : llm_call / eval / bootstrap / feed_assembly events
    """

    def __init__(self, out_dir, jsonl_name: str = 'timing.jsonl',
                 status_name: str = 'STATUS.md'):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / jsonl_name
        self.status_path = self.out_dir / status_name
        self._t0 = time.perf_counter()
        # Touch the JSONL so it exists from the start (post-mortem grep is
        # easier when the path is always real).
        if not self.jsonl_path.exists():
            self.jsonl_path.write_text('')

    def record(self, kind: str, event: str, **fields: Any) -> None:
        """Append one JSONL line with timestamp + elapsed-since-init.

        `kind` is the granularity tag ('phase' / 'iteration' / 'substep').
        `event` is the specific event name (e.g. 'phase_start', 'llm_call').
        """
        rec = {
            'ts':              datetime.now(timezone.utc).isoformat(),
            'elapsed_total_s': round(time.perf_counter() - self._t0, 3),
            'kind':            kind,
            'event':           event,
        }
        rec.update(fields)
        with open(self.jsonl_path, 'a', encoding='utf-8') as f:
            f.write(json.dumps(rec, default=str) + '\n')

    @contextmanager
    def time(self, kind: str, event: str, **fields: Any) -> Iterator[Dict[str, Any]]:
        """Context manager that emits two records around the with-block:
        a `<event>__start` and a `<event>__end` with the duration filled in.

        Yields a mutable dict you can mutate inside the block; whatever you
        add gets attached to the `__end` record. Useful for emitting
        per-call token counts, gen latency, etc.
        """
        start = time.perf_counter()
        self.record(kind, f'{event}__start', **fields)
        scratch: Dict[str, Any] = {}
        try:
            yield scratch
        finally:
            seconds = time.perf_counter() - start
            payload = dict(fields)
            payload.update(scratch)
            payload['seconds'] = round(seconds, 4)
            self.record(kind, f'{event}__end', **payload)

    def update_status(self, status: Dict[str, Any]) -> None:
        """Rewrite STATUS.md with a flat human-readable dashboard.

        `status` is a dict { phase_name: state-or-metadata }. State strings
        like 'pending' / 'running' / 'complete' / 'failed' get a friendly
        prefix; everything else is dumped verbatim as JSON.
        """
        lines = ['# Round 1 — STATUS',
                 '',
                 f'_last update: {datetime.now(timezone.utc).isoformat()}_',
                 '',
                 f'_elapsed since logger init: '
                 f'{round(time.perf_counter() - self._t0, 1)} s_',
                 '',
                 '| phase | state | notes |',
                 '|-------|-------|-------|']
        glyph = {'pending':  '⋯',
                 'running':  '▶',
                 'complete': '✓',
                 'failed':   '✗',
                 'skipped':  '–'}
        for phase, val in status.items():
            if isinstance(val, str):
                state = val
                notes = ''
            elif isinstance(val, dict):
                state = str(val.get('state', '?'))
                notes = json.dumps({k: v for k, v in val.items() if k != 'state'},
                                    default=str)
            else:
                state = '?'; notes = json.dumps(val, default=str)
            mark = glyph.get(state, '?')
            lines.append(f'| {phase} | {mark} {state} | {notes} |')
        lines.append('')
        lines.append(f'_jsonl source of truth: `{self.jsonl_path.name}`_')
        self.status_path.write_text('\n'.join(lines) + '\n', encoding='utf-8')
