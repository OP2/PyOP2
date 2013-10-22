# Calculate an optimisation plan for a list of kernels

from pyop2.ir.ast_optimiser import LoopOptimiser
from pyop2.ir.ast_vectoriser import LoopVectoriser
from pyop2.ir.ast_base import *


class KernelPlan(object):

    """Optimise a kernel. """

    def __init__(self, kernel_ast):
        self.kernel_ast = kernel_ast
        self.decl, self.fors, self.dlabs = self._visit_ir(kernel_ast)

    def _visit_ir(self, node, fors=[], dlabs=[], decls={}):
        """Return lists of:
            - declarations within the kernel
            - perfect loop nests
            - dense linear algebra blocks
        that will be exploited at plan creation time."""

        if isinstance(node, Decl):
            decls[node.sym.symbol] = node
            return (decls, fors, dlabs)
        if isinstance(node, For):
            fors.append(node)
            return (decls, fors, dlabs)
        if isinstance(node, Statement) and node.pragma:
            dlabs.append(node)
            return (decls, fors, dlabs)

        for c in node.children:
            self._visit_ir(c, fors, dlabs)

        return (decls, fors, dlabs)

    def plan_cpu(self, isa, compiler):
        lo = [LoopOptimiser(fors) for fors in self.fors]

        for nest in lo:
            # Loop optimisations
            nest.licm()

            # Update declarations due to licm
            self.decl.update(nest.decls)

            # Vectorisation
            vect = LoopVectoriser(self.kernel_ast, nest, isa, compiler)
            vect.pad_and_align(self.decl)
            vect.adjust_loop(False)
            vect.set_alignment(self.decl, True)
            vect.outer_product()
