def normalize_api_base_url(base_url: str, keep: bool = False) -> str:
    """Normalize an API URL to end exactly with ``/v1``."""
    if keep:
        return base_url
    normalized = base_url.strip().rstrip("/")
    if "/v1" in normalized:
        normalized = normalized.split("/v1", 1)[0]
    return f"{normalized}/v1"
