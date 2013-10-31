# Calculate an optimisation plan for a list of kernels

from pyop2.ir.ast_optimiser import LoopOptimiser
from pyop2.ir.ast_vectoriser import LoopVectoriser
from pyop2.ir.ast_base import *


class KernelPlan(object):

    """Optimise a kernel. """

    def __init__(self, kernel_ast):
        self.kernel_ast = kernel_ast
        self.decl, self.fors, self.dlabs = self._visit_ir(kernel_ast)

    def _visit_ir(self, node, parent=None, fors=[], dlabs=[], decls={}):
        """Return lists of:
            - declarations within the kernel
            - perfect loop nests
            - dense linear algebra blocks
        that will be exploited at plan creation time."""

        if isinstance(node, Decl):
            decls[node.sym.symbol] = node
            return (decls, fors, dlabs)
        if isinstance(node, For):
            fors.append((node, parent))
            return (decls, fors, dlabs)
        if isinstance(node, Statement) and node.pragma:
            dlabs.append(node)
            return (decls, fors, dlabs)

        for c in node.children:
            self._visit_ir(c, node, fors, dlabs, decls)

        return (decls, fors, dlabs)

    def plan_cpu(self, isa, compiler, opts):
        """Optimize the kernel for cpu execution. """

        # Fetch user-provided options/hints on how to transform the kernel
        perm = opts["interchange"]
        licm = opts["licm"]
        layout = opts["pad_and_align"]
        out_vect = opts["outer-product tiling"]

        lo = [LoopOptimiser(l, pre_l) for l, pre_l in self.fors]
        for nest in lo:
            # 1) Loop interchange
            if perm:
                nest.interchange(perm)

            # 2) Loop invariant code motion
            if licm:
                inv_outer_loops = nest.licm()
                self.decl.update(nest.decls)
                # Optimisation of invariant loops
                for l in inv_outer_loops:
                    opt_l = LoopOptimiser(l, self.kernel_ast)
                    vect = LoopVectoriser(opt_l, isa, compiler)
                    vect.adjust_loop(True)
                    #vect.set_alignment(self.decl, True)

            vect = LoopVectoriser(nest, isa, compiler)

            # 3) Padding and data alignment
            if layout:
                vect.pad_and_align(self.decl)
                vect.adjust_loop(True)
                vect.set_alignment(self.decl, True)

            # 4) Outer-product vectorisation
            if out_vect in [1, 2, 3]:
                vect.outer_product(out_vect)
            elif out_vect == 4:
                nest.tiling_outer_product()
