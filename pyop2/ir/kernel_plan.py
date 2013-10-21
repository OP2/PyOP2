# Calculate an optimisation plan for a list of kernels

from pyop2.ir.ast_optimiser import LoopOptimiser
from pyop2.ir.ast_vectoriser import LoopVectoriser
from pyop2.ir.ast_base import *


class KernelPlan(object):

    """Optimise a kernel. """

    def __init__(self, kernel_ast):
        self.kernel_ast = kernel_ast
        self.fors, self.dlabs = self._visit_ir(kernel_ast)

    def create_plan(self, backend="sequential"):
        if backend == "sequential":
            self._plan_cpu()

    def _visit_ir(self, node, fors=[], dlabs=[]):
        """Return lists of:
            - perfect loop nests
            - dense linear algebra blocks
        that will be exploited at plan creation time."""

        if isinstance(node, For):
            fors.append(node)
            return (fors, dlabs)
        if isinstance(node, Statement) and node.pragma:
            dlabs.append(node)
            return (fors, dlabs)

        for c in node.children:
            self._visit_ir(c, fors, dlabs)

        return (fors, dlabs)

    def _plan_cpu(self):
        lo = [LoopOptimiser(fors) for fors in self.fors]

        for nest in lo:
            # Loop optimisations
            nest.licm()

            # Vectorisation
            # FIXME: backend
            vect = LoopVectoriser(self.kernel_ast, nest, "AVX")
            vect.pad_and_align()
            vect.adjust_loop()
