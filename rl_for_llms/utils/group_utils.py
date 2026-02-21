from collections.abc import Generator


def iter_groups[T](
    items: list[T], group_size: int
) -> Generator[tuple[int, int, list[T]]]:
    """Yield ``(start, end, group_slice)`` for consecutive groups of *group_size*."""
    num_groups = len(items) // group_size
    for group_idx in range(num_groups):
        start = group_idx * group_size
        end = start + group_size
        yield start, end, items[start:end]
