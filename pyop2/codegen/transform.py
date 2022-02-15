"""Kernel transforms."""


def memoize_maps(kernel):
    """Lift indirect loops to memoized maps passed in as arguments."""
    return kernel.copy(...)
