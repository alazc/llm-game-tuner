"""Prompts package — versioned prompt artefacts + hash-lock loader.

`from prompts.loader import load_prompt` to load a prompt with hash verification.
"""
from prompts.loader import load_prompt, load_prompt_with_meta, PromptHashMismatch

__all__ = ['load_prompt', 'load_prompt_with_meta', 'PromptHashMismatch']
