"""Replace entire function/method bodies in installed vLLM source files."""

from __future__ import annotations

from pathlib import Path


class PatchError(SystemExit):
    pass


def replace_whole_body(
    path: Path,
    upstream_path: Path,
    patched_path: Path,
    label: str,
    *,
    skip_if_upstream_missing: bool = False,
) -> bool:
    """Replace one function body. Returns True if the target file was modified."""
    text = path.read_text()
    upstream = upstream_path.read_text()
    patched = patched_path.read_text()

    if upstream == patched:
        print(f"{label}: skipped (upstream and patched are identical; no patch needed)")
        return False

    if patched in text:
        print(f"{label}: already patched")
        return False

    if upstream not in text:
        if skip_if_upstream_missing:
            print(f"{label}: skipped (upstream not found; likely already in vLLM)")
            return False
        raise PatchError(
            f"Patch failed: {label} upstream body not found "
            f"(vLLM version may have changed; update {upstream_path})"
        )

    path.write_text(text.replace(upstream, patched, 1))
    print(f"{label}: replaced entire body")
    return True
