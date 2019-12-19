from pyop2.utils import cached_property
from itertools import count


def noop_decorate(fn):
    def wrapper(self):
        return fn(self)
    return wrapper


class Cached(object):
    def __init__(self):
        self.a = count()
        self.b = count()

    @cached_property
    def undecorated(self):
        return next(self.a)

    @cached_property
    @noop_decorate
    def decorated(self):
        return next(self.b)


def test_cached_property():
    obj = Cached()

    assert obj.undecorated == 0
    assert obj.undecorated == 0
    assert obj.decorated == 0
    assert obj.decorated == 0
