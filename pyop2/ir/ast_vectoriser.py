from math import ceil
from copy import deepcopy as dcopy

from pyop2.ir.ast_base import *


class LoopVectoriser(object):

    """ Loop vectorizer
        * Vectorization:
          Outer-product vectorisation.
        * Memory:
          padding, data alignment, trip count/bound adjustment
          padding and data alignment are for aligned unit-stride load
          trip count/bound adjustment is for auto-vectorisation. """

    def __init__(self, kernel_ast, loop_optimiser, isa, compiler):
        self.lo = loop_optimiser
        self.ast = kernel_ast
        self.intr = self._set_isa(isa)
        self.comp = self._set_compiler(compiler)
        self.i_loops = []
        self._inner_loops(self.lo.loop_nest, self.i_loops)

    # Memory optimisations #

    def pad_and_align(self, decl):
        """Pad each data structure accessed in the loop nest to a multiple
        of the vector length. """

        # Get the declarations of the symbols accessed in the loop nest
        acc_decl = [d for s, d in decl.items() if s in self.lo.sym]

        # Do the rounding of the inner dimension (padding)
        for d in acc_decl:
            rounded = self._roundup(d.sym.rank[-1])
            d.sym.rank = d.sym.rank[:-1] + (rounded,)

    def adjust_loop(self, only_bound):
        """Adjust trip count and bound of each innermost loop in the nest."""

        for l in self.i_loops:
            # Also adjust the loop increment factor
            if not only_bound:
                l.incr.children[1] = c_sym(self.intr["dp_reg"])

            # Adjust the loops bound
            bound = l.cond.children[1]
            l.cond.children[1] = c_sym(self._roundup(bound.symbol))

    def set_alignment(self, decl, autovectorisation=False):
        """Align arrays in the kernel to the size of the vector length in
        order to issue aligned loads and stores. Also tell this information to
        the back-end compiler by adding suitable pragmas over loops in case
        we rely on autovectorisation. """

        for d in decl.values():
            d.attr.append(self.comp["align"](self.intr["alignment"]))

        if autovectorisation:
            for l in self.i_loops:
                l.pragma = self.comp["decl_aligned_for"]

    # Vectorisation

    def outer_product(self, opts=None):
        """Compute outer products according to opts. """

        for stmt, loops in self.lo.out_prods.items():
            op = OuterProduct(stmt, loops, self.intr)
            body, layout = op.generate()
            self.lo.block.children = body
            # Append the layout code after the outer product loops
            # TODO: now appending *after* the loop nest, can instead be done
            # by appending at a certain depth depending on which loops (outer
            # product loops vs other loops) come first
            parent = self.lo.pre_header.children
            parent.insert(parent.index(self.lo.loop_nest) + 1, layout)

    def peel(self):
        """Peel iterations out of the outer_product loops s.t. the iteration
        space becomes rectancular with sides of length multiple of the vector
        length (e.g. in the case of AVX, 4x4, 8x4, 8x8, ...), and create
        additional loops for the remainder of the iteration space. """

    # Utilities
    def _inner_loops(self, node, loops):
        """Find the inner loops in the tree rooted in node."""
        if perf_stmt(node):
            return False
        elif isinstance(node, Block):
            return any([self._inner_loops(s, loops) for s in node.children])
        elif isinstance(node, For):
            found = self._inner_loops(node.children[0], loops)
            if not found:
                loops.append(node)
            return True

    def _set_isa(self, isa):
        """Set the proper intrinsics instruction set. """

        if isa == "AVX":
            return {
                "inst_set": "AVX",
                "avail_reg": 16,
                "alignment": 32,
                "dp_reg": 4,  # Number of double values per register
                "reg": lambda n: "ymm%s" % n,
                "zeroall": "_mm256_zeroall ()",
                "setzero": "_mm256_setzero_pd ()",
                "decl_var": "__m256d",
                "align_array": lambda p: "__attribute__((aligned(%s)))" % p,
                "symbol": lambda s, r: AVXLoad(s, r),
                "store": lambda m, r: AVXStore(m, r),
                "mul": lambda r1, r2: AVXProd(r1, r2),
                "div": lambda r1, r2: AVXDiv(r1, r2),
                "add": lambda r1, r2: AVXSum(r1, r2),
                "sub": lambda r1, r2: AVXSub(r1, r2),
                "l_perm": lambda r, f: AVXLocalPermute(r, f),
                "g_perm": lambda r1, r2, f: AVXGlobalPermute(r1, r2, f),
                "unpck_hi": lambda r1, r2: AVXUnpackHi(r1, r2),
                "unpck_lo": lambda r1, r2: AVXUnpackLo(r1, r2)
            }

    def _set_compiler(self, compiler):
        """Set compiler-specific keywords. """

        if compiler == "INTEL":
            return {
                "align": lambda o: "__attribute__((aligned(%s)))" % o,
                "decl_aligned_for": "#pragma vector aligned"
            }

    def _roundup(self, x):
        """Return x rounded up to the vector length. """
        word_len = self.intr["dp_reg"]
        return int(ceil(x / float(word_len))) * word_len


class OuterProduct(object):

    """Compute outer product vectorisation of a statement. """

    def __init__(self, stmt, loops, intr):
        self.stmt = stmt
        self.loops = loops
        self.intr = intr

    class Alloc(object):

        """Handle allocation of register variables. """

        def __init__(self, intr, factor):
            # TODO Use factor when unrolling...
            nres = intr["dp_reg"]
            self.ntot = intr["avail_reg"]
            self.res = [intr["reg"](v) for v in range(nres)]
            self.var = [intr["reg"](v) for v in range(nres, self.ntot)]
            self.i = intr

        def get_reg(self):
            if len(self.var) == 0:
                l = self.ntot * 2
                self.var += [self.i["reg"](v) for v in range(self.ntot, l)]
                self.ntot = l
            return self.var.pop(0)

        def free_reg(self, reg):
            self.var.insert(0, reg)

        def get_tensor(self):
            return self.res

    def _swap_reg(self, step, reg):
        """Swap values in a vector register. """

        if step == 0:
            return []
        elif step in [1, 3]:
            return [self.intr["l_perm"](r, "5") for r in reg.values()]
        elif step == 2:
            return [self.intr["g_perm"](r, r, "1") for r in reg.values()]

    def _vect_mem(self, node, regs, loops, decls=[], in_vrs={}, out_vrs={}):
        """Return a list of vector variables declarations representing
        loads, sets, broadcasts. Also return dicts of allocated inner
        and outer variables. """

        if isinstance(node, Symbol):
            # Check if this symbol's values are iterated over with the
            # fastest varying dimension of the loop nest
            reg = regs.get_reg()
            if node.rank and loops[1] == node.rank[-1]:
                in_vrs[node] = Symbol(reg, ())
            else:
                out_vrs[node] = Symbol(reg, ())
            expr = self.intr["symbol"](node.symbol, node.rank)
            decls.append(Decl(self.intr["decl_var"], Symbol(reg, ()), expr))
        elif isinstance(node, Par):
            child = node.children[0]
            self._vect_mem(child, regs, loops, decls, in_vrs, out_vrs)
        else:
            left = node.children[0]
            right = node.children[1]
            self._vect_mem(left, regs, loops, decls, in_vrs, out_vrs)
            self._vect_mem(right, regs, loops, decls, in_vrs, out_vrs)

        return (decls, in_vrs, out_vrs)

    def _vect_expr(self, node, in_vrs, out_vrs):
        """Turn a scalar expression into its intrinsics equivalent. """

        if isinstance(node, Symbol):
            return in_vrs.get(node) or out_vrs.get(node)
        elif isinstance(node, Par):
            return self._vect_expr(node.children[0], in_vrs, out_vrs)
        else:
            left = self._vect_expr(node.children[0], in_vrs, out_vrs)
            right = self._vect_expr(node.children[1], in_vrs, out_vrs)
            if isinstance(node, Sum):
                return self.intr["add"](left, right)
            elif isinstance(node, Sub):
                return self.intr["sub"](left, right)
            elif isinstance(node, Prod):
                return self.intr["mul"](left, right)
            elif isinstance(node, Div):
                return self.intr["div"](left, right)

    def _incr_tensor(self, tensor, ofs, out_reg, mode):
        """Add the right hand side contained in out_reg to tensor."""
        if mode == 0:
            # Store in memory
            loc = (tensor.rank[0] + "+" + str(ofs), tensor.rank[1])
            return self.intr["store"](Symbol(tensor.symbol, loc), out_reg)

    def _restore_layout(self, regs, tensor, mode):
        """Restore the storage layout of the tensor. """

        # Determine tensor symbols
        tensor_syms = []
        for i in range(self.intr["dp_reg"]):
            rank = (tensor.rank[0] + "+" + str(i), tensor.rank[1])
            tensor_syms.append(Symbol(tensor.symbol, rank))

        code = []
        t_regs = [Symbol(r, ()) for r in regs.get_tensor()]

        # Load LHS values from memory
        for i, j in zip(tensor_syms, t_regs):
            load_sym = self.intr["symbol"](i.symbol, i.rank)
            code.append(Decl(self.intr["decl_var"], j, load_sym))

        # In-register restoration of the tensor
        # TODO: AVX only at the present moment
        perm = self.intr["g_perm"]
        uphi = self.intr["unpck_hi"]
        uplo = self.intr["unpck_lo"]
        typ = self.intr["decl_var"]
        n_reg = self.intr["dp_reg"]
        if mode == 0:
            tmp = [Symbol(regs.get_reg(), ()) for i in range(n_reg)]
            code.append(Decl(typ, tmp[0], uphi(t_regs[1], t_regs[0])))
            code.append(Decl(typ, tmp[1], uplo(t_regs[0], t_regs[1])))
            code.append(Decl(typ, tmp[2], uphi(t_regs[2], t_regs[3])))
            code.append(Decl(typ, tmp[3], uplo(t_regs[3], t_regs[2])))
            code.append(Assign(t_regs[0], perm(tmp[1], tmp[3], 32)))
            code.append(Assign(t_regs[1], perm(tmp[0], tmp[2], 32)))
            code.append(Assign(t_regs[2], perm(tmp[3], tmp[1], 49)))
            code.append(Assign(t_regs[3], perm(tmp[2], tmp[0], 49)))

        # Store LHS values in memory
        for i, j in zip(tensor_syms, t_regs):
            code.append(self.intr["store"](i, j))

        return code

    def generate(self):
        # TODO: need to determine order of loops w.r.t. the local tensor
        # entries. E.g. if j-k inner loops and A[j][k], then increments of
        # A are performed within the k loop. On the other hand, if ip is
        # the innermost loop, stores in memory are done outside of ip
        mode = 0  # 0 == Stores, 1 == Local incrs

        loops_it = tuple([l.it_var() for l in self.loops])
        tensor = self.stmt.children[0]
        expr = self.stmt.children[1]

        # Get source-level variables
        regs = self.Alloc(self.intr, 1)  # TODO: set appropriate factor

        # Find required loads
        decls, in_vrs, out_vrs = self._vect_mem(expr, regs, loops_it)
        stmt = []
        for i in range(self.intr["dp_reg"]):
            # Register shuffles, vectorisation of a row, update tensor
            stmt.extend(self._swap_reg(i, in_vrs))
            intr_expr = self._vect_expr(expr, in_vrs, out_vrs)
            stmt.append(self._incr_tensor(tensor, i, intr_expr, mode))
        # Restore the tensor layout
        layout = self._restore_layout(regs, tensor, mode)

        # Create the vectorized for body
        new_block = []
        for d in decls + stmt:
            new_block.append(d)

        # Code for restoring the layout
        layout_loops = dcopy(self.loops)
        layout_loops[0].children = [Block([layout_loops[1]], open_scope=True)]
        layout_loops[1].children = [Block(layout, open_scope=True)]
        return (new_block, layout_loops[0])
