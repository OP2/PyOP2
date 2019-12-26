import loopy as lp
import numpy as np
import pycuda.driver as cuda
from math import ceil, sqrt, floor
from pytools import memoize_method
from pycuda.compiler import SourceModule
from pyop2.utils import cached_property

WARP_SIZE = 32


class GreedyKernelGenerator:
    def __init__(self, fem_program, num_candidate_knls):
        self.fem_program = fem_program
        self.num_candidate_knls = num_candidate_knls

    @cached_property
    def nbasis(self):
        #FIXME: Not sure if this will hold all the times
        return int(lp.symbolic.pw_aff_to_expr(
            self.fem_program.root_kernel.get_iname_bounds('form_i',
                constants_only=True).size))

    @cached_property
    def nquad(self):
        #FIXME: Not sure if this will hold all the times
        return int(lp.symbolic.pw_aff_to_expr(
                self.fem_program.root_kernel.get_iname_bounds('form_ip',
                    constants_only=True).size))

    @cached_property
    def num_matrices(self):
        #FIXME: Nauseating naming
        const_matrices_in_quad = set()
        const_matrices_in_basis = set()
        const_matrices = frozenset([tv.name for tv in
            self.fem_program.root_kernel.temporary_variables.values() if
            tv.initializer is not None and len(tv.initializer.shape) == 2])

        for insn in self.fem_program.root_kernel.instructions:
            if 'quadrature' in insn.tags:
                const_matrices_in_quad.update(insn.read_dependency_names() &
                        const_matrices)
            if 'basis' in insn.tags:
                const_matrices_in_basis.update(insn.read_dependency_names() &
                        const_matrices)

        return max(len(const_matrices_in_quad), len(const_matrices_in_basis))

    @cached_property
    def num_func_eval_vars(self):
        #FIXME: Terrible naming
        evaluation_variables = (set().union(*[insn.write_dependency_names() for
            insn in self.fem_program.root_kernel.instructions if 'quadrature' in insn.tags]) &
            set().union(*[insn.read_dependency_names() for insn in
                self.fem_program.root_kernel.instructions if 'basis' in insn.tags]))

        return len(evaluation_variables)

    def get_local_barriers(self, t1_r, t1_c, t2_r, t2_c):
        return (
                ceil(self.nquad/t1_r) * ceil(self.nbasis/t1_c)
                + ceil(self.nbasis/t2_r) * ceil(self.nquad/t2_c))

    def theoretical_warps_per_sm(self, cells_per_block,
            threads_per_cell, t1_r, t1_c, t2_r, t2_c):

        # {{{ computing shared mem usage per block

        shared_usage = (
                self.num_matrices*max(t1_r*t1_c, t2_r*t2_c)
                + self.nquad
                + self.num_func_eval_vars*self.nquad*cells_per_block
                )

        # convert doubles to KB
        shared_usage *= 8e-3

        # }}}

        warps_per_block = floor((threads_per_cell*cells_per_block)/32)
        blocks_per_sm = min(96//shared_usage if shared_usage < 48 else 0, 32)
        warps_per_sm = blocks_per_sm*warps_per_block

        return warps_per_sm

    def get_work_efficiency(self, cells_per_block,
            threads_per_cell, t1_r, t1_c, t2_r, t2_c):

        # wasted work in the function evaluation stage
        wasted_work = self.nbasis*(
                (t1_r % threads_per_cell)*(self.nquad//t1_r)
                + ((self.nquad % t1_r) % threads_per_cell))

        wasted_work += self.nquad*(
                (t2_r % threads_per_cell)*(self.nbasis//t2_r)
                + ((self.nbasis % t2_r) % threads_per_cell))

        wasted_work_fraction = wasted_work / (2*self.nquad*self.nbasis)

        threads_in_block = threads_per_cell * cells_per_block
        warp_mismatch_factor = threads_in_block / (
                threads_in_block + (WARP_SIZE - (threads_in_block % WARP_SIZE)))

        return warp_mismatch_factor*(1-wasted_work_fraction)

    def actual_warps_per_sm(self, cells_per_block, threads_per_cell,
            t1_r, t1_c, t2_r, t2_c):
        return (
                self.theoretical_warps_per_sm(cells_per_block,
                    threads_per_cell, t1_r, t1_c, t2_r, t2_c)
                * self.get_work_efficiency(cells_per_block,
                threads_per_cell, t1_r, t1_c, t2_r, t2_c))

    @memoize_method
    def estimated_exec_time(self, cells_per_block, threads_per_cell,
            t1_r, t1_c, t2_r, t2_c):
        nb, nq = self.nbasis, self.nquad
        n_w = self.actual_warps_per_sm(cells_per_block,
                threads_per_cell, t1_r, t1_c, t2_r, t2_c)
        if n_w == 0:
            return float("inf")
        n_lb = self.get_local_barriers(t1_r, t1_c, t2_r, t2_c)
        n_t = threads_per_cell
        n_c = cells_per_block
        n_blocks = (n_w * 32)/(n_t*n_c)

        return n_lb/n_blocks
        return n_lb/n_w

        return (n_t*nb + nb*nq/(n_t*n_c) + nb*nq*(n_t+n_c)/20.0)/n_w

    def heuristic_param_space(self):

        threads_to_cells = {
                9: (7, ),
                8: (4, 8, 16),
                7: (9, ),
                4: (8, 16),
                3: (21, ),
                2: (16, 32, 64),
                1: (32, 64),
                }

        tiles = []

        for i in range(1, ceil(sqrt(self.nbasis))+1):
            t1_c = ceil(self.nbasis/i)
            for j in range(1, ceil(sqrt(self.nquad))+1):
                t1_r = ceil(self.nquad/j)
                for k in range(1, ceil(sqrt(self.nbasis))+1):
                    t2_r = ceil(self.nbasis/k)
                    for l in range(1, ceil(sqrt(self.nquad))+1):
                        t2_c = ceil(self.nquad/l)
                        if abs(t1_r*t1_c-t2_r*t2_c)/max(t1_r*t1_c, t2_c*t2_r) < 0.2:
                            tiles.append((t1_r, t1_c, t2_r, t2_c))

        # sort by least sync-ed config first
        tiles.sort(key=lambda T: self.get_local_barriers(*T))

        params = []

        for tile in tiles:
            for threads in threads_to_cells:
                best_cells = 10000
                for cells in threads_to_cells[threads]:
                    if (self.estimated_exec_time(cells, threads,
                        *tile) < self.estimated_exec_time(best_cells,
                            threads, *tile)):
                        best_cells = cells

                if best_cells != 10000:
                    params.append((best_cells, threads)+tile)

        # sort the parameters with highest occupancy.
        params.sort(key=lambda P:  self.estimated_exec_time(*P))

        return params[:self.num_candidate_knls]

    @memoize_method
    def convert_numpy_arrays_to_cuda_mems(self, ary):
        ary = np.array(ary)
        ary_gpu = cuda.mem_alloc(ary.nbytes)
        cuda.memcpy_htod(src=ary, dest=ary_gpu)
        return ary_gpu

    def __call__(self, args, argshapes):
        best_performing_time = float("inf")
        best_performing_param = None
        nrounds = 15
        nwarmup = 5

        # TODO: Need to make copies of the data given by the user.
        # for getting the copies we somehow need the sizes of the arrays. :o
        # not sure how do we get the data
        copied_args = args[:2]
        for i, arg in enumerate(self.fem_program.args[2:]):
            if arg.name in self.fem_program.root_kernel.get_written_variables():
                arg_gpu = cuda.mem_alloc(
                        int(np.prod(argshapes[i])*arg.dtype.itemsize))
                cuda.memcpy_dtod(src=args[i+2], dest=arg_gpu,
                        size=int(np.prod(argshapes[i])*arg.dtype.itemsize))
                copied_args += (arg_gpu,)
            else:
                copied_args += (args[i+2],)

        from pyop2.gpu.tile import tiled_transform

        for params in self.heuristic_param_space():
            nc, nt, t1_r, t1_c, t2_r, t2_c = params
            kernel, extra_args = tiled_transform(
                    self.fem_program.root_kernel, self.fem_program.callables_table,
                    nc, nt, t1_r, t1_c, t2_r, t2_c, False, False, True, True,
                    False, False)
            from pymbolic import evaluate
            kernel = self.fem_program.with_root_kernel(kernel)
            code = lp.generate_code_v2(kernel).device_code()
            glens, llens = kernel.get_grid_size_upper_bounds_as_exprs()
            grid = tuple(int(evaluate(glens[i], {"start": args[0], "end":
                args[1]})) if i < len(glens) else 1
                    for i in range(2))
            block = tuple(int(evaluate(llens[i], {"start": args[0], "end":
                args[1]})) if i < len(llens) else 1
                    for i in range(3))
            executable_knl = SourceModule(code).get_function(kernel.name)
            executable_knl.prepare("i"*2+"P"*len(args[2:])+"P"*len(extra_args))
            extra_args = tuple(self.convert_numpy_arrays_to_cuda_mems(tuple(arg)) for arg
                    in extra_args)
            runtimes = []

            for i in range(nrounds):
                start_evt = cuda.Event()
                end_evt = cuda.Event()
                start_evt.record()
                start_evt.synchronize()
                executable_knl.prepared_call(grid, block, *(copied_args+extra_args))
                end_evt.record()
                end_evt.synchronize()
                runtimes.append(start_evt.time_till(end_evt)/1000)

            exec_time = np.mean(runtimes[nwarmup:])

            print("Params: {}, time={}".format(
                params, exec_time))

            if exec_time < best_performing_time:
                best_performing_time = exec_time
                best_performing_param = params

        nc, nt, t1_r, t1_c, t2_r, t2_c = best_performing_param
        return tiled_transform(
                self.fem_program.root_kernel, self.fem_program.callables_table,
                nc, nt, t1_r, t1_c, t2_r, t2_c, False, False, True, True,
                False, False)
