class DescMap(object):
    def __init__(self, name):
        self._name = name
        self._entries = dict()

    def __setitem__(self, key, value):
        self._entries[str(key)] = str(value)

    def __str__(self):
        desc = "%s\n" % self._name
        for (key, value) in sorted(self._entries.iteritems()):
            keyline = ("  " + key + ": ")
            width = len(keyline)
            for value_line in value.split("\n"):
                desc += keyline + value_line + "\n"
                keyline = (" " * width)
        return desc.rstrip("\n")

    def _max_key_length(self):
        return reduce(lambda x, y: max(x, len(y)), self._entries.keys(), 0)
