"""Hash-locked prompt loader.

Every system prompt used by a round-1 experiment is stored as a `.txt` file
under `prompts/`, paired with a `.json` sidecar that records the prompt's
metadata and a SHA256 of the `.txt` contents. `load_prompt` reads the text,
recomputes the hash, compares against the sidecar, and raises if they differ.

Why: a silent edit to a system prompt mid-experiment would invalidate every
trajectory recorded against the old version. The sidecar makes that drift
loud — the loader aborts the run before the first LLM call.

Update procedure when a prompt legitimately changes:
  1. Edit the `.txt`.
  2. Run `python -m prompts.loader --rehash <path>` to refresh the sidecar.
  3. Commit both files together so the diff is auditable.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Tuple


PROMPTS_DIR = Path(__file__).resolve().parent


def compute_sha256(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _sidecar_path(txt_path: Path) -> Path:
    return txt_path.with_suffix('.json')


class PromptHashMismatch(RuntimeError):
    """Raised when a prompt's on-disk text hash differs from its sidecar's
    recorded hash. Means the prompt was edited without refreshing the sidecar
    (or the sidecar was edited without the text)."""


def load_prompt(path) -> str:
    """Read the prompt text at `path`, verify it against its `.json` sidecar's
    `sha256` field, and return the verified text.

    Raises:
      FileNotFoundError - txt or sidecar missing.
      PromptHashMismatch - text hash differs from sidecar's recorded hash.
    """
    txt = Path(path)
    if not txt.is_absolute():
        # Resolve relative paths against the prompts directory so callers can
        # write `load_prompt('character_llm_prompt.txt')` regardless of cwd.
        candidate = PROMPTS_DIR / txt
        txt = candidate if candidate.exists() else txt
    if not txt.exists():
        raise FileNotFoundError(f'prompt text not found: {txt}')
    sidecar = _sidecar_path(txt)
    if not sidecar.exists():
        raise FileNotFoundError(f'prompt sidecar not found: {sidecar}')
    text = txt.read_text(encoding='utf-8')
    meta = json.loads(sidecar.read_text(encoding='utf-8'))
    expected = meta.get('sha256')
    actual = compute_sha256(text)
    if expected != actual:
        raise PromptHashMismatch(
            f'{txt.name}: sha256 mismatch.\n'
            f'  sidecar  ({sidecar.name}) records: {expected}\n'
            f'  text     ({txt.name})    hashes to: {actual}\n'
            f'Either revert the edit or rerun: python -m prompts.loader '
            f'--rehash {txt}'
        )
    return text


def load_prompt_with_meta(path) -> Tuple[str, dict]:
    """Same as `load_prompt` but also returns the sidecar metadata dict."""
    text = load_prompt(path)
    sidecar = _sidecar_path(Path(path) if Path(path).is_absolute()
                            else (PROMPTS_DIR / path))
    meta = json.loads(sidecar.read_text(encoding='utf-8'))
    return text, meta


def rehash(txt_path) -> dict:
    """Recompute the sidecar `sha256` for `txt_path` from the current text.
    Returns the updated sidecar dict. Used by the CLI rehash hook below."""
    txt = Path(txt_path)
    if not txt.exists():
        raise FileNotFoundError(txt)
    sidecar = _sidecar_path(txt)
    if sidecar.exists():
        meta = json.loads(sidecar.read_text(encoding='utf-8'))
    else:
        meta = {'name': txt.stem}
    text = txt.read_text(encoding='utf-8')
    meta['sha256'] = compute_sha256(text)
    sidecar.write_text(json.dumps(meta, indent=2) + '\n', encoding='utf-8')
    return meta


def _cli() -> int:
    ap = argparse.ArgumentParser(description='Prompt hash-lock CLI')
    ap.add_argument('--rehash', metavar='TXT_PATH',
                    help='Recompute the sha256 in the sidecar for the given '
                         '.txt path (use after deliberately editing a prompt).')
    ap.add_argument('--verify', metavar='TXT_PATH',
                    help='Verify a prompt loads cleanly. Exits non-zero on '
                         'hash mismatch.')
    args = ap.parse_args()
    if args.rehash:
        meta = rehash(args.rehash)
        print(f'rehashed {args.rehash} -> {meta["sha256"]}')
        return 0
    if args.verify:
        load_prompt(args.verify)
        print(f'OK: {args.verify}')
        return 0
    ap.print_help()
    return 2


if __name__ == '__main__':
    raise SystemExit(_cli())
