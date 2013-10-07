# Calculate an optimisation plan for a list of kernels

from ast_optimiser import *


class KernelPlan(object):

    def __init__(self, kernels_ir):
        self.kernels_ir = kernels_ir

    def plan(self, backend="sequential"):
        if backend == "sequential":
            pass

    def _plan_cpu(self):
        pass
