import enum


# Don't like this. Seems very hacky.
class Relation(enum.Enum):
    DIRECT = enum.auto()
    CLOSURE = enum.auto()

    def __mul__(self, other):
        if isinstance(other, Relation):
            return self, other
        else:
            return NotImplemented


def as_tag(relation):
    """throws KeyError if not found"""
    return {Relation.DIRECT: "direct",
            Relation.CLOSURE: "closure"}[relation]
