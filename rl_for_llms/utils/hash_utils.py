import hashlib
import json
import typing


def generate_deterministic_id(
    data_dict: dict[str, typing.Any],
    id_string_length: int = 16,
) -> str:
    """Generate a deterministic ID based on the input data."""
    string_representation = json.dumps(data_dict)
    hash_value = hashlib.blake2b(
        string_representation.encode(), digest_size=id_string_length // 2
    ).hexdigest()
    return hash_value
