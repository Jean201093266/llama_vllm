"""Shared CLI helpers."""

from __future__ import annotations

from llama_vllm.config.preflight import PreflightValidationError


def format_auto_fix_message(exc: PreflightValidationError, *, show_raw: bool = False) -> str:
    """Format preflight error with numbered remediation commands."""
    lines = ["Preflight validation failed:"]
    for item in exc.errors:
        lines.append(f"- {item}")

    if exc.formatted_suggestions:
        lines.append("Recommended fix command (1):")
        lines.append(f"  1) {exc.formatted_suggestions[0]}")
        if len(exc.formatted_suggestions) > 1:
            lines.append("Other fix commands:")
            for idx, cmd in enumerate(exc.formatted_suggestions[1:], start=2):
                lines.append(f"  {idx}) {cmd}")

    if show_raw and exc.suggestions:
        lines.append("Raw override suggestions:")
        for idx, item in enumerate(exc.suggestions, start=1):
            lines.append(f"  {idx}) {item}")

    return "\n".join(lines)


def apply_first_suggestion(existing_overrides: list[str], suggestions: list[str]) -> list[str]:
    """Apply the first suggestion override and return merged overrides."""
    if not suggestions:
        return list(existing_overrides)

    first = suggestions[0].strip()
    if first.startswith("--override "):
        first = first[len("--override ") :]

    def _key(item: str) -> str:
        if "=" not in item:
            return item.strip()
        return item.split("=", 1)[0].strip()

    first_key = _key(first)
    merged = [item for item in existing_overrides if _key(item) != first_key]
    merged.append(first)
    return merged


