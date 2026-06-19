"""Domain entity tagging for regulation chunks."""

from src.models.graph import NodeLabel

KNOWN_CONCEPTS: dict[str, NodeLabel] = {
    "fcr": NodeLabel.BALANCING_PRODUCT,
    "afrr": NodeLabel.BALANCING_PRODUCT,
    "mfrr": NodeLabel.BALANCING_PRODUCT,
    "imbalance settlement": NodeLabel.CONCEPT,
    "balancing reserve": NodeLabel.CONCEPT,
    "capacity allocation": NodeLabel.CONCEPT,
    "day-ahead": NodeLabel.CONCEPT,
    "network code": NodeLabel.CONCEPT,
}

KNOWN_NETWORK_CODES: dict[str, str] = {
    "so gl": "SO GL",
    "eb gl": "EB GL",
    "nc rfg": "NC RfG",
    "nc cacm": "NC CACM",
}


def find_known_terms(text: str) -> list[tuple[str, NodeLabel]]:
    """Return domain entity names found in regulation text."""
    lowered = text.lower()
    found: list[tuple[str, NodeLabel]] = []
    seen: set[str] = set()

    for term, label in sorted(KNOWN_CONCEPTS.items(), key=lambda item: -len(item[0])):
        if term in lowered and term not in seen:
            found.append((term.upper() if len(term) <= 4 else term.title(), label))
            seen.add(term)

    for term, display in sorted(
        KNOWN_NETWORK_CODES.items(), key=lambda item: -len(item[0])
    ):
        if term in lowered and display not in seen:
            found.append((display, NodeLabel.NETWORK_CODE))
            seen.add(display)

    return found


def entity_names_for_chunk(text: str) -> list[str]:
    """Entity display names to store in chunk metadata."""
    return [name for name, _ in find_known_terms(text)]
