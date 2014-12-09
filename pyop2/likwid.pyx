cdef extern from "likwid.h":
    extern void likwid_markerInit()
    extern void likwid_markerClose()
    extern void likwid_markerThreadInit()


def initialise():
    likwid_markerInit()
    likwid_markerThreadInit()


def finalise():
    likwid_markerClose()