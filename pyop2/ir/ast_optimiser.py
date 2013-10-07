# Loops optimiser: licm, register tiling, unroll-and-jam, peeling
#   licm is usually about moving stuff independent of the inner-most loop
#       here, a slightly different algorithm is employed: only const values are
#       searched in a statement (i.e. read-only values), but their motion
#       takes into account the whole loop nest. Therefore, this is licm
#       tailored to assembly routines
#   register tiling
#   unroll-and-jam
#   peeling
# Memory optimiser: padding, data alignment, trip count/bound adjustment
#   padding and data alignment are for aligned unit-stride load
#   trip count/bound adjustment is for auto-vectorisation

from ast_base import *


class LoopOptimiser(object):

    """Loops optimiser: licm, register tiling, unroll-and-jam, peeling ."""

    def __init__(self, loop_nest):
        self.loop_nest = loop_nest

    def licm(self):
        print "\nFinding loop-invariant code"
        embed()
