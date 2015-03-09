cdef extern from "papi.h":
    extern void PAPI_init_librart(PAPI_VERSION_CURRENT)


def initialise():
    likwid_markerInit(PAPI_VERSION_CURRENT)
