from __future__ import annotations


FORMAT_VERSION = "life_maybe_format_v1"


def _clean(text: str) -> str:
    return (
        text.replace("_", " ")
        .replace("-LRB-", "(")
        .replace("-RRB-", ")")
        .replace("-COLON-", ":")
    )


def life_format_evidence(items: list[str]) -> str:
    formatted = []
    for item in items:
        if "\t" not in item:
            formatted.append(item)
            continue
        title, content = item.split("\t", 1)
        formatted.append(f"title: {_clean(title)} context: {_clean(content)}")
    return " ### ".join(formatted)
