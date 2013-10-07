# Calculate an optimisation plan for a list of kernels

from pyop2.ir.ast_base import For, Statement


class KernelPlan(object):

    """Optimise a kernel. """

    def __init__(self, kernels_ir):
        self.kernels_ir = kernels_ir
        self._for, self._dlabs = self._visit_ir(kernels_ir)

    def create_plan(self, backend="sequential"):
        if backend == "sequential":
            self._plan_cpu()

    def _visit_ir(self, node, _for=[], _dlabs=[]):
        """Return lists of:
            - (perfect) loop nests
            - dense linear algebra blocks
        that will be exploited at plan creation time."""

        if isinstance(node, For):
            _for.append(node)
            return (_for, _dlabs)
        if isinstance(node, Statement) and node.pragma:
            _dlabs.append(node)
            return (_for, _dlabs)

        for c in node.children:
            self._visit_ir(c, _for, _dlabs)

        return (_for, _dlabs)

    def _plan_cpu(self):
        pass
