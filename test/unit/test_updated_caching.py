import pytest
from functools import partial

from pyop2.caching import (  # noqa: F401
    disk_only_cache,
    memory_cache,
    memory_and_disk_cache,
    default_parallel_hashkey,
    clear_memory_cache
)
from pyop2.mpi import MPI, COMM_WORLD, comm_cache_keyval  # noqa: F401


class StateIncrement:
    """Simple class for keeping track of the number of times executed
    """
    def __init__(self):
        self._count = 0

    def __call__(self):
        self._count += 1
        return self._count

    @property
    def value(self):
        return self._count


def twople(x):
    return (x, )*2


def threeple(x):
    return (x, )*3


def n_comms(n):
    return [MPI.COMM_WORLD]*n


def n_ops(n):
    return [MPI.SUM]*n


# decorator = parallel_memory_only_cache, parallel_memory_only_cache_no_broadcast, disk_only_cached
def function_factory(state, decorator, f, **kwargs):
    def custom_function(x, comm=COMM_WORLD):
        state()
        return f(x)

    return decorator(**kwargs)(custom_function)


@pytest.fixture
def state():
    return StateIncrement()


@pytest.mark.parametrize("decorator, uncached_function", [
    (memory_cache, twople),
    (partial(memory_cache, broadcast=False), n_comms),
    (memory_and_disk_cache, twople),
    (disk_only_cache, twople)
])
def test_function_args_twice_caches(request, state, decorator, uncached_function, tmpdir):
    if request.node.callspec.params["decorator"] in {disk_only_cache, memory_and_disk_cache}:
        kwargs = {"cachedir": tmpdir}
    else:
        kwargs = {}

    cached_function = function_factory(state, decorator, uncached_function, **kwargs)
    assert state.value == 0
    first = cached_function(2, comm=COMM_WORLD)
    assert first == uncached_function(2)
    assert state.value == 1
    second = cached_function(2, comm=COMM_WORLD)
    assert second == uncached_function(2)
    if request.node.callspec.params["decorator"] is not disk_only_cache:
        assert second is first
    assert state.value == 1

    clear_memory_cache(COMM_WORLD)


@pytest.mark.parametrize("decorator, uncached_function", [
    (memory_cache, twople),
    (partial(memory_cache, broadcast=False), n_comms),
    (memory_and_disk_cache, twople),
    (disk_only_cache, twople)
])
def test_function_args_different(request, state, decorator, uncached_function, tmpdir):
    if request.node.callspec.params["decorator"] in {disk_only_cache, memory_and_disk_cache}:
        kwargs = {"cachedir": tmpdir}
    else:
        kwargs = {}

    cached_function = function_factory(state, decorator, uncached_function, **kwargs)
    assert state.value == 0
    first = cached_function(2, comm=COMM_WORLD)
    assert first == uncached_function(2)
    assert state.value == 1
    second = cached_function(3, comm=COMM_WORLD)
    assert second == uncached_function(3)
    assert state.value == 2

    clear_memory_cache(COMM_WORLD)


@pytest.mark.parallel(nprocs=3)
@pytest.mark.parametrize("decorator, uncached_function", [
    (memory_cache, twople),
    (partial(memory_cache, broadcast=False), n_comms),
    (memory_and_disk_cache, twople),
    (disk_only_cache, twople)
])
def test_function_over_different_comms(request, state, decorator, uncached_function, tmpdir):
    if request.node.callspec.params["decorator"] in {disk_only_cache, memory_and_disk_cache}:
        # In parallel different ranks can get different tempdirs, we just want one
        tmpdir = COMM_WORLD.bcast(tmpdir, root=0)
        kwargs = {"cachedir": tmpdir}
    else:
        kwargs = {}

    cached_function = function_factory(state, decorator, uncached_function, **kwargs)
    assert state.value == 0

    for ii in range(10):
        color = 0 if COMM_WORLD.rank < 2 else MPI.UNDEFINED
        comm12 = COMM_WORLD.Split(color=color)
        if COMM_WORLD.rank < 2:
            _ = cached_function(2, comm=comm12)
            comm12.Free()

        color = 0 if COMM_WORLD.rank > 0 else MPI.UNDEFINED
        comm23 = COMM_WORLD.Split(color=color)
        if COMM_WORLD.rank > 0:
            _ = cached_function(2, comm=comm23)
            comm23.Free()

    clear_memory_cache(COMM_WORLD)
