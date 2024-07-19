import loopy as lp
import numpy as np
import pycuda.driver as cuda
from math import ceil, sqrt, floor
from pytools import memoize_method
from pycuda.compiler import SourceModule
from pyop2.utils import cached_property
from pytools import memoize_on_first_arg
import dataclasses as dc
from abc import ABC, abstractmethod
import pycuda.gpuarray as cuda_np
from typing import FrozenSet, List, Tuple, Sequence, Union
from functools import reduce as ft_reduce
from loopy.symbolic import (
    IdentityMapper,
    RuleAwareIdentityMapper,
    SubstitutionRuleMappingContext,
)
import pymbolic.primitives as prim


# {{{ Modeling a transform candidate.


@dc.dataclass(frozen=True, init=False)
class TransformCandidate(ABC):
    @abstractmethod
    def __init__(self):
        pass


@dc.dataclass(frozen=True)
class SWIPC(TransformCandidate):
    """
    Single Work-item per Cell transformation.
    """


@dc.dataclass(frozen=True)
class ParametricTiling(TransformCandidate):
    """
    Records the configuration for :func:`pyop2.gpu.tile.tiled_transform`.

    :attr ncells_per_block: Number of cells whose computation workload is to be
        given to one CUDA block.
    :attr nthreads_per_cell: Number of CUDA threads to be launched for one each
        cell in the mesh.
    :attr matvec1_row_tile_length: Number of rows in the tile of the first
        matvec (first matvec := quadrature stage)
    :attr matvec1_col_tile_length: Number of columns in the tile of the first
        matvec (first matvec := quadrature stage)
    :attr matvec2_row_tile_length: Number of rows in the tile of the second
        matvec (second matvec := output DoF stage)
    :attr matvec2_col_tile_length: Number of columns in the tile of the second
        matvec (second matvec := output DoF stage)
    :attr load_coordinates_to_shared: Should the coordinates of the cell be
        prefetched to shared memory?
    :attr load_input_to_shared: Should the input DoFs be prefetched to shared
        memory?
    :attr load_mats_to_shared: Should the local FEM operator matrices be loaded
        to shared memory?
    :attr load_quad_weights_to_shared: Should the quadrature weights be loaded
        to shared memory?
    :attr tiled_prefetch_of_inputs: If input DoFs are prefetched to shared
        memory, should they be prefetched in tile lengths?
    :attr tiled_prefetch_of_quad_weights: If the quadrature weights are
        prefetched to shared memory, should they in prefetched in tile lengths?
    """

    ncells_per_block: int
    nthreads_per_cell: int
    operator_tile_descriptions: Tuple[int, Tuple[int, ...]]
    quad_rowtile_lengths: Tuple[int, ...]
    load_coordinates_to_shared: bool
    load_input_to_shared: bool
    load_mats_to_shared: bool
    load_quad_weights_to_shared: bool
    tiled_prefetch_of_inputs: bool
    tiled_prefetch_of_quad_weights: bool


# }}}

# {{{ loopy transform helpers


class AxesRemovingMapper(IdentityMapper):
    def __init__(self, tv_to_removable_axes):
        self.tv_to_removable_axes = tv_to_removable_axes
        super().__init__()

    def map_subscript(self, expr):
        removable_indices = self.tv_to_removable_axes.get(expr.aggregate.name, None)

        if removable_indices:
            assert all(expr.index_tuple[idx] == 0 for idx in removable_indices)
            new_expr = prim.Subscript(
                expr.aggregate,
                tuple(
                    self.rec(idx)
                    for i, idx in enumerate(expr.index_tuple)
                    if i not in removable_indices
                ),
            )

            return new_expr

        return super(AxesRemovingMapper, self).map_subscript(expr)


def remove_unused_axes_in_temporaries(kernel):
    new_temps = {}
    tv_x_removable_axes = {}
    for tv in kernel.temporary_variables.values():
        removable_axes = tuple(
            i for i, axis_len in enumerate(tv.shape) if axis_len == 1
        )
        if removable_axes:
            tv_x_removable_axes[tv.name] = removable_axes
            new_temps[tv.name] = tv.copy(
                shape=tuple(axis_len for axis_len in tv.shape if axis_len != 1),
                dim_tags=None,
            )
        else:
            new_temps[tv.name] = tv

    new_insns = []
    axes_removing_mapper = AxesRemovingMapper(tv_x_removable_axes)

    for insn in kernel.instructions:
        new_insns.append(insn.with_transformed_expressions(axes_removing_mapper))

    return kernel.copy(instructions=new_insns, temporary_variables=new_temps)


class AxisRemover(RuleAwareIdentityMapper):
    def __init__(self, rule_mapping_context, var_name, axis_num):
        self.var_name = var_name
        self.axis_num = axis_num
        super(AxisRemover, self).__init__(rule_mapping_context)

    def map_subscript(self, expr, expn_state):
        if expr.aggregate.name == self.var_name:
            if len(expr.index_tuple) == 1:
                return prim.Variable(self.var_name)
            else:
                return prim.Subscript(
                    expr.aggregate,
                    expr.index_tuple[: self.axis_num]
                    + expr.index_tuple[self.axis_num + 1 :],
                )

        return super(AxisRemover, self).map_subscript(expr, expn_state)


def remove_axis(kernel, var_name, axis_num):
    assert var_name in kernel.temporary_variables

    assert axis_num < len(kernel.temporary_variables[var_name].shape)

    rule_mapping_context = SubstitutionRuleMappingContext(
        kernel.substitutions, kernel.get_var_name_generator()
    )

    kernel = AxisRemover(rule_mapping_context, var_name, axis_num).map_kernel(kernel)

    if len(kernel.temporary_variables[var_name].shape) == 1:
        new_temps = dict(
            (
                (tv.name, tv.copy(shape=(), dim_tags=None))
                if tv.name == var_name
                else (tv.name, tv)
            )
            for tv in kernel.temporary_variables.values()
        )
    else:
        from loopy import auto

        new_temps = dict(
            (
                (
                    tv.name,
                    tv.copy(
                        shape=tv.shape[:axis_num] + tv.shape[axis_num + 1 :],
                        strides=auto,
                        dim_tags=None,
                    ),
                )
                if tv.name == var_name
                else (tv.name, tv)
            )
            for tv in kernel.temporary_variables.values()
        )

    return kernel.copy(temporary_variables=new_temps)


def remove_invariant_inames(kernel):
    inames_used = set()
    untagged_inames = kernel.all_inames() - frozenset(kernel.iname_to_tags.keys())
    for insn in kernel.instructions:
        for iname in (
            insn.read_dependency_names() | insn.write_dependency_names()
        ) & untagged_inames:
            inames_used.add(iname)

    removable_inames = untagged_inames - inames_used

    new_insns = [
        insn.copy(within_inames=insn.within_inames - removable_inames)
        for insn in kernel.instructions
    ]

    from loopy.transform.inames import remove_unused_inames

    return remove_unused_inames(
        kernel.copy(instructions=new_insns), removable_inames
    )


# }}}


# {{{ implementing the tiling transformation


@lp.for_each_kernel
def remove_unnecessary_deps(kernel):

    from loopy.schedule import get_insns_in_topologically_sorted_order

    insn_order = get_insns_in_topologically_sorted_order(kernel)

    new_insns = insn_order.copy()

    for i, source_insn in enumerate(insn_order):
        if isinstance(source_insn, lp.MultiAssignmentBase):
            written_var_name = source_insn.assignee_name

            for j, sink_insn in enumerate(insn_order[i + 1 :]):
                if written_var_name in sink_insn.read_dependency_names():
                    assert new_insns[j + i + 1].id == sink_insn.id
                    new_insns[j + 1 + i] = new_insns[j + 1 + i].copy(
                        depends_on=(
                            new_insns[j + 1 + i].depends_on
                            | frozenset([source_insn.id])
                        )
                    )
                else:
                    assert new_insns[j + i + 1].id == sink_insn.id
                    new_insns[j + 1 + i] = new_insns[j + 1 + i].copy(
                        depends_on=(
                            new_insns[j + 1 + i].depends_on
                            - frozenset([source_insn.id])
                        )
                    )

    return kernel.copy(instructions=new_insns)


def find_recursive_reverse_dependencies(
    kernel: lp.LoopKernel, insn_ids: FrozenSet[str]
) -> FrozenSet[str]:
    assert isinstance(insn_ids, frozenset)
    all_rev_depends = set()
    from loopy.kernel.tools import find_reverse_dependencies

    new_rev_depends = insn_ids

    while new_rev_depends:
        new_rev_depends = (
            find_reverse_dependencies(kernel, new_rev_depends)
            - all_rev_depends
            - insn_ids
        )
        all_rev_depends |= new_rev_depends

    return frozenset(all_rev_depends)


def _make_tv_array_arg(tv):
    assert tv.address_space != lp.AddressSpace.PRIVATE
    arg = lp.ArrayArg(
        name=tv.name,
        dtype=tv.dtype,
        shape=tv.shape,
        dim_tags=tv.dim_tags,
        offset=tv.offset,
        dim_names=tv.dim_names,
        order=tv.order,
        alignment=tv.alignment,
        address_space=tv.address_space,
        is_input=tv.read_only,
        is_output=not tv.read_only,
    )
    return arg


@dc.dataclass(frozen=True)
class MatvecStageDescr:
    dof_names: Tuple[str, ...]
    row_iname: str
    col_iname: str
    deriv_matrices: FrozenSet[str]

    copy = dc.replace


@dc.dataclass(frozen=True, kw_only=True)
class KernelMetadata:
    iquad: str
    coords: str
    trialDoF_gather_inames: List[str]  # noqa: N815
    outDoF_init_iname: str  # noqa: N815
    quad_weights: str
    matvec_stage_descrs: List[MatvecStageDescr]
    eval_results: FrozenSet[str]
    scatter_iname: str
    n_trial_derivs: List[int]
    n_quad: int
    n_outDoF: int  # noqa: N815
    n_trialDoFs: List[int]  # noqa: N815

    @property
    def outDoF(self) -> str:
        """
        Returns the output DOF name in the FEM action kernel *kernel*.
        """
        return self.matvec_stage_descrs[-1].dof_names[0]

    @property
    def n_trial_stages(self):
        return len(self.matvec_stage_descrs) - 1


def are_mv_stages_similar(mv_stage_x, mv_stage_y):
    return (mv_stage_x.deriv_matrices == mv_stage_y.deriv_matrices) and (
        mv_stage_x.col_iname == mv_stage_x.col_iname
    )


@memoize_on_first_arg
def temp_vars_both_read_and_write_access(
    kernel: lp.LoopKernel, insn: lp.InstructionBase
):
    tvs = frozenset(kernel.temporary_variables)
    read_tvs = insn.read_dependency_names() & tvs
    write_tvs = insn.write_dependency_names() & tvs
    return read_tvs & write_tvs


def inference_which_should_ideally_be_done_by_passing_metadata(kernel):
    """
    Only intended to work for the vanilla representation of the form kernel.
    For ex. Sum factorized action kernels won"t fit the pattern.
    """
    icell = "n"

    # quad iname
    # Assumption: There is only a single iname responsible for quadrature and
    # it starts with "form_ip">
    (iquad,) = [
        iname for iname in kernel.all_inames() if iname.startswith("form_ip")
    ]

    # trialDof_x_outputDofs_x_coords: A set containing the variable names for the
    # *temporaries* of trialDofs, outputDofs and the coordinates.
    # trialDof, outputDofs, coords := local DoFs
    # These are also the variables which are written (or initialized during the
    # gather phase).
    trialDofs_x_outDof_x_coords = (
        ft_reduce(
            frozenset.union,
            (
                insn.write_dependency_names()
                for insn in kernel.instructions
                if lp.match.Tagged("gather")(kernel, insn)
            ),
            frozenset(),
        )
        - kernel.all_inames()
    )

    # {{{ extract outputDoF name

    # Assumption: There is only one output DoF being generated in our 1-form
    # assembly.
    # In the "quadr" phase of the form kernel the *only* variable being written
    # is output DoF

    (outDoF,) = {
        insn.assignee_name
        for insn in kernel.instructions
        if lp.match.Tagged("quadrature")(kernel, insn)
    }

    # }}}

    # {{{ extract coords name

    # Assumptions: coordinate transformation is affine i.e. one
    # Jacobian computation for each cell. Thereby all the instructions
    # responsible for computing entries of the Jacobian matrix would be only
    # within the "n" loop.
    coords = set()
    for insn in kernel.instructions:
        if lp.match.Tagged("evaluate")(kernel, insn) and (
            insn.within_inames == frozenset(["n"])
        ):
            coords = coords | (
                insn.read_dependency_names() & trialDofs_x_outDof_x_coords
            )

    (coords,) = coords

    # }}}

    # {{{ extract trial DoF names

    trialDoFs = trialDofs_x_outDof_x_coords - frozenset([coords, outDoF])

    # }}}

    # {{{ scatter iname

    # Logic: Already assumed that there is only one outDof pet kernel; so
    # picking up the scatter insn based on that singleton variable.

    (scatter_insn,) = [
        insn
        for insn in kernel.instructions
        if lp.match.Tagged("scatter")(kernel, insn)
    ]
    scatter_map = scatter_insn.assignee.index_tuple[0]
    (scatter_iname,) = set(scatter_map.index_tuple) - set([prim.Variable("n")])
    scatter_iname = scatter_iname.name

    # }}}

    # {{{ output DoF init iname

    # Assumption the outDoF init instruction is as follows:
    # outDoF[outDoF_init_iname, ...] <- 0

    (outDoF_init_iname,) = [
        insn.assignee.index_tuple[1].name
        for insn in kernel.instructions
        if (
            lp.match.Tagged("gather")(kernel, insn)
            and (outDoF == insn.assignee_name)
        )
    ]

    # }}}

    # {{{ doF_inames_in_eval_stage

    # dof_inames_in_eval_stage: iname corresponding the reduction loop in the
    # eval matvecs. These inames have been represented by $i_1$, $i_2$, ... in the
    # paper.
    doF_inames_in_eval_stage = set()
    trialDofs_to_redn_inames = {}
    for trialDoF in trialDoFs:
        # trialDoF is read only in the accumulate instruction in the matvec of
        # the eval stage and the instruction accessing would it have the
        # inames: "n, iquad, i_1".  Over here we extract what"s the name of i_1
        # in our FEM kernel.
        (iname,) = ft_reduce(
            set.union,
            (
                insn.within_inames
                for insn in kernel.instructions
                if trialDoF in insn.read_dependency_names()
            ),
            set(),
        ) - {"n", iquad}

        doF_inames_in_eval_stage.add(iname)
        trialDofs_to_redn_inames[trialDoF] = iname

    # }}}

    # {{{ tagging the stages of the kernel

    new_insns = []

    for insn in kernel.instructions:
        if lp.match.Tagged("gather")(kernel, insn):
            fem_action_phase = "gather"
        elif insn.within_inames == frozenset([icell]):
            fem_action_phase = "jacobi"
        elif (
            insn.within_inames == frozenset([icell, iquad])
            and insn.expression == 0
            and lp.match.Tagged("evaluate")(kernel, insn)
        ):
            fem_action_phase = "eval_init"
        elif insn.within_inames == frozenset([icell, iquad]) and lp.match.Tagged(
            "evaluate"
        )(kernel, insn):
            fem_action_phase = "eval_wrap_up"
        elif (
            insn.within_inames > frozenset([icell, iquad])
            and temp_vars_both_read_and_write_access(kernel, insn)
            and lp.match.Tagged("evaluate")(kernel, insn)
        ):
            fem_action_phase = "eval_redn"
        elif lp.match.Tagged("quadrature")(kernel, insn):
            assert temp_vars_both_read_and_write_access(kernel, insn)
            fem_action_phase = "quadr_redn"
        elif lp.match.Tagged("scatter")(kernel, insn):
            fem_action_phase = "quadr_wrap_up"
        else:
            print("Failed for -- ", insn)
            raise NotImplementedError(insn)

        new_insns.append(
            insn.tagged(lp.LegacyStringInstructionTag(fem_action_phase))
        )

    kernel = kernel.copy(instructions=new_insns)

    # }}}

    # {{{ extract deriv_matrices, quad_weights

    # derivative matrices are the constant data whose array dimensions > 1
    deriv_matrices = {
        tv.name
        for tv in kernel.temporary_variables.values()
        if tv.initializer is not None and len(tv.initializer.shape) != 1
    }

    # quad_weights is the only constant data in the kernel which is a single
    # dimensional array
    (quad_weights,) = [
        tv.name
        for tv in kernel.temporary_variables.values()
        if tv.initializer is not None and len(tv.initializer.shape) == 1
    ]
    # }}}

    matvec_descrs = []

    # {{{ identify matvec stages for the eval-part of the compute kernel

    from loopy.match import parse_match

    for i, (trialDoF, redn_iname) in enumerate(trialDofs_to_redn_inames.items()):
        within = parse_match("writes:%s" % trialDoF)
        (trialDof_init_insn_id,) = [
            insn.id for insn in kernel.instructions if within(kernel, insn)
        ]

        # all the recursive reverse dependencies of trialDoF_init_insn in the
        # eval-part of the kernel form the trialDoF"s matvec

        matvec_insn_ids = find_recursive_reverse_dependencies(
            kernel, frozenset({trialDof_init_insn_id})
        )
        kernel = lp.tag_instructions(
            kernel,
            "matvec%d" % i,
            "("
            + " or ".join(
                ["id:%s" % matvec_insn_id for matvec_insn_id in matvec_insn_ids]
            )
            + ") and (tag:eval_init or tag:eval_redn)",
        )

        vars_written_in_matvec = ft_reduce(
            frozenset.union,
            (
                insn.write_dependency_names()
                for insn in kernel.instructions
                if lp.match.Tagged(f"matvec{i}")(kernel, insn)
            ),
            frozenset(),
        )
        eval_init_insn_ids = [
            insn.id
            for insn in kernel.instructions
            if (
                (insn.assignee_name in vars_written_in_matvec)
                and lp.match.Tagged("eval_init")(kernel, insn)
            )
        ]

        kernel = lp.tag_instructions(
            kernel,
            f"matvec{i}",
            " or ".join(
                [
                    "id:%s" % eval_init_insn_id
                    for eval_init_insn_id in eval_init_insn_ids
                ]
            ),
        )

        deriv_matrices_in_current_mv_stg = (
            ft_reduce(
                frozenset.union,
                (
                    insn.read_dependency_names()
                    for insn in kernel.instructions
                    if lp.match.Tagged(f"matvec{i}")(kernel, insn)
                ),
                frozenset(),
            )
            & deriv_matrices
        )

        matvec_descrs.append(
            MatvecStageDescr(
                (trialDoF,), iquad, redn_iname, deriv_matrices_in_current_mv_stg
            )
        )

    # }}}

    # {{{ extract matvec producing outDoF

    ((quadr_stage_DoF_iname,),) = {
        (insn.within_inames - {"n", iquad})
        for insn in kernel.instructions
        if lp.match.Tagged("quadrature")(kernel, insn)
    }

    kernel = lp.tag_instructions(
        kernel,
        f"matvec{i+1}",
        "(tag:gather or tag:quadrature) and (reads:{0} or writes:{0})".format(
            outDoF
        ),  # noqa: E501
    )
    kernel = lp.tag_instructions(
        kernel, "quadr_init", f"tag:gather and tag:matvec{i+1}"
    )

    deriv_matrices_in_current_mv_stg = (
        ft_reduce(
            frozenset.union,
            (
                insn.read_dependency_names()
                for insn in kernel.instructions
                if lp.match.Tagged(f"matvec{i+1}")(kernel, insn)
            ),
            frozenset(),
        )
        & deriv_matrices
    )

    matvec_descrs.append(
        MatvecStageDescr(
            (outDoF,), quadr_stage_DoF_iname, iquad, deriv_matrices_in_current_mv_stg
        )
    )

    # }}}

    # eval_results: temporary variables which are the final result of the
    # evaluation part of the kernel.
    # Hence, eval_results =Variables which are written in the eval stage and
    # read in the quadr stage
    eval_results = frozenset().union(
        *[
            insn.write_dependency_names()
            for insn in kernel.instructions
            if lp.match.Tagged("eval_wrap_up")(kernel, insn)
        ]
    ) & frozenset().union(
        *[
            insn.read_dependency_names()
            for insn in kernel.instructions
            if lp.match.Tagged("quadrature")(kernel, insn)
        ]
    )

    # {{{ fuse matvec stages

    # to_be_fused_mv_stages: list of tuples of MV stages which are to be fused.
    to_be_fused_mv_stages = []

    for i, mv_stage_i in enumerate(matvec_descrs):
        if any(
            (i, mv_stage_i) in to_be_fused_stage
            for to_be_fused_stage in to_be_fused_mv_stages
        ):
            continue
        to_be_fused_mv_stage = ((i, mv_stage_i),)
        # do not fuse "quadr" stage matvec with any other matvec
        for j, mv_stage_j in enumerate(matvec_descrs[i + 1 : -1], start=i + 1):
            if are_mv_stages_similar(mv_stage_i, mv_stage_j):
                to_be_fused_mv_stage = to_be_fused_mv_stage + ((j, mv_stage_j),)

        to_be_fused_mv_stages.append(to_be_fused_mv_stage)

    mv_stage_descrs_post_fusion = []

    for to_be_fused_mv_stage in to_be_fused_mv_stages:
        current_mv_stg_idx = len(mv_stage_descrs_post_fusion)

        def retag_insn(insn):
            new_tags = {
                tag
                for tag in insn.tags
                if not (isinstance(tag, lp.LegacyStringInstructionTag)
                        and tag.value.startswith("matvec"))
            } | {lp.LegacyStringInstructionTag(f"matvec{current_mv_stg_idx}")}
            return insn.copy(tags=frozenset(new_tags))

        kernel = lp.map_instructions(
            kernel,
            " or ".join("tag:matvec%d" % i for i, _ in to_be_fused_mv_stage),
            retag_insn,
        )
        fused_dof_names = tuple(
            mv_stg.dof_names[0] for _, mv_stg in to_be_fused_mv_stage
        )
        new_mv_stage = to_be_fused_mv_stage[0][1].copy(dof_names=fused_dof_names)
        mv_stage_descrs_post_fusion.append(new_mv_stage)

    # }}}

    # {{{ trialDoF gather iname

    # Assumption the trialDoF gather instruction is as follows:
    # trialDoF[trialDoF_gather_iname, ...] <- datxx[mapxx[trialDof_gather_iname, ...], ...]  # noqa: E501

    trialDoF_to_gather_inames = {}
    trialDoF_gather_inames = []
    for trialDoF in trialDoFs:
        (trialDoF_gather_iname,) = [
            insn.assignee.index_tuple[1].name
            for insn in kernel.instructions
            if (trialDoF == insn.assignee_name)
        ]
        trialDoF_to_gather_inames[trialDoF] = trialDoF_gather_iname

    for mv_stage in mv_stage_descrs_post_fusion[:-1]:
        fused_trialDoF_gather_iname = trialDoF_to_gather_inames[
            mv_stage.dof_names[0]
        ]
        for trialDoF in mv_stage.dof_names[1:]:
            if trialDoF_to_gather_inames[trialDoF] == fused_trialDoF_gather_iname:
                continue
            kernel = lp.rename_iname(
                kernel,
                trialDoF_to_gather_inames[trialDoF],
                fused_trialDoF_gather_iname,
                existing_ok=True,
            )

        trialDoF_gather_inames.append(fused_trialDoF_gather_iname)

    # }}}

    n_trial_derivs = [
        len(
            [
                insn
                for insn in kernel.instructions
                if lp.match.And(
                    (lp.match.Tagged("matvec%d" % i), lp.match.Tagged("eval_init"))
                )(kernel, insn)
            ]
        )
        for i, _ in enumerate(trialDoFs)
    ]

    n_quad = int(
        lp.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds(iquad, constants_only=True).size
        )
    )
    n_outDoF = int(
        lp.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds(
                mv_stage_descrs_post_fusion[-1].row_iname, constants_only=True
            ).size
        )
    )

    n_trialDoFs = [
        int(
            lp.symbolic.pw_aff_to_expr(
                kernel.get_iname_bounds(
                    mv_stg_descr.col_iname, constants_only=True
                ).size
            )
        )
        for mv_stg_descr in mv_stage_descrs_post_fusion[:-1]
    ]

    return kernel, KernelMetadata(
        iquad=iquad,
        coords=coords,
        outDoF_init_iname=outDoF_init_iname,
        quad_weights=quad_weights,
        matvec_stage_descrs=mv_stage_descrs_post_fusion,
        scatter_iname=scatter_iname,
        eval_results=eval_results,
        trialDoF_gather_inames=trialDoF_gather_inames,
        n_trial_derivs=n_trial_derivs,
        n_quad=n_quad,
        n_outDoF=n_outDoF,
        n_trialDoFs=n_trialDoFs,
    )


def tiled_transform(t_unit, tiling_config):
    """
    :param tiling_config: An instance of :class:`pyop2.gpu.tiling_config
    """

    kernel = t_unit.default_entrypoint
    callables_table = t_unit.callables_table
    assert isinstance(kernel, lp.LoopKernel)
    assert isinstance(tiling_config, ParametricTiling)

    # {{{ Inferring variables

    kernel, metadata = inference_which_should_ideally_be_done_by_passing_metadata(
        kernel
    )
    iquad = metadata.iquad
    coords = metadata.coords
    outDoF = metadata.outDoF
    outDoF_init_iname = metadata.outDoF_init_iname
    scatter_iname = metadata.scatter_iname
    quad_weights = metadata.quad_weights
    matvec_stage_descrs = metadata.matvec_stage_descrs
    eval_results = metadata.eval_results
    nquad = metadata.n_quad
    n_outDoF = metadata.n_outDoF
    n_trialDoFs = metadata.n_trialDoFs
    n_trial = metadata.n_trial_stages
    trialDoF_gather_inames = metadata.trialDoF_gather_inames

    # }}}

    nc = tiling_config.ncells_per_block
    nt = tiling_config.nthreads_per_cell
    mv_tiles = tiling_config.operator_tile_descriptions
    quad_tiles = tiling_config.quad_rowtile_lengths

    if mv_tiles == ():
        mv_tiles = tuple((nquad, nDoF) for nDoF in n_trialDoFs) + (
            (n_outDoF, nquad),
        )
    if quad_tiles == ():
        quad_tiles = (nquad,)
    (quad_tile,) = quad_tiles

    assert all(len(tile) == 2 for tile in mv_tiles)
    assert len(mv_tiles) == len(matvec_stage_descrs)  # one for each mv stage
    assert (
        len({mv_tile[0] for mv_tile in mv_tiles[:-1]}) == 1
    )  # in the general case only one $T_e^r$ is supported

    T_e_r = mv_tiles[0][0]
    T_e_cs = [tile[1] for tile in mv_tiles[:-1]]
    T_q_r = mv_tiles[-1][0]
    T_q_c = mv_tiles[-1][1]

    kernel = lp.split_iname(kernel, iquad, quad_tile, outer_iname="iquad_tile")
    kernel = lp.rename_iname(kernel, iquad + "_inner", iquad)

    # {{{ privatize temps for function evals and make them LOCAL

    kernel = lp.privatize_temporaries_with_inames(kernel, iquad, eval_results)

    kernel = lp.set_temporary_scope(kernel, eval_results, lp.AddressSpace.LOCAL)

    # }}}

    # {{{ Duplicate inames to separate transformation logic for different matvecs

    for i, mv_stg_descr in enumerate(matvec_stage_descrs):
        kernel = lp.duplicate_inames(
            kernel, mv_stg_descr.col_iname, "tag:matvec%d" % i, "icol%d" % i
        )

    kernel = lp.duplicate_inames(kernel, iquad, "tag:eval", "irow_eval")
    kernel = lp.duplicate_inames(
        kernel, matvec_stage_descrs[-1].row_iname, "tag:quadrature", "irow_quadr"
    )

    # }}}

    # {{{ change address space of constants to "__global"

    old_temps = kernel.temporary_variables.copy()
    args_to_make_global = [
        tv.initializer.flatten()
        for tv in old_temps.values()
        if tv.initializer is not None
    ]

    new_temps = dict(
        (tv.name, tv) for tv in old_temps.values() if tv.initializer is None
    )
    kernel = kernel.copy(
        args=kernel.args
        + [
            _make_tv_array_arg(tv)
            for tv in old_temps.values()
            if tv.initializer is not None
        ],
        temporary_variables=new_temps,
    )

    # }}}

    from loopy.loop import merge_loop_domains

    kernel = merge_loop_domains(kernel)
    kernel = remove_unused_axes_in_temporaries(kernel)

    # Realize CUDA blocks
    kernel = lp.split_iname(
        kernel, "n", nc, outer_iname="iblock", inner_iname="icell"
    )

    # Privatize eval_results
    kernel = lp.privatize_temporaries_with_inames(
        kernel, "icell", only_var_names=eval_results
    )

    # cut down the size of the number of basis coeffs written by each
    # thread(if there are multiple threads)
    kernel = lp.rename_iname(kernel, scatter_iname, "irow_quadr", True)
    kernel = lp.rename_iname(kernel, outDoF_init_iname, "irow_quadr", True)

    kernel = remove_axis(kernel, outDoF, 0)

    # enfoce dependency of first matvec stage onto the jacobian evaluation stage
    kernel = lp.add_dependency(kernel, "tag:eval_init and tag:matvec0", "tag:jacobi")

    # {{{ prefetch coordinates (not implemented)

    if tiling_config.load_coordinates_to_shared:
        # FIXME: This configuration parameter seems unnecessary as of now. I
        # might choose not to support it.
        kernel = lp.privatize_temporaries_with_inames(kernel, "icell", [coords])
        kernel = lp.assignment_to_subst(kernel, coords)
        raise NotImplementedError("This might be only useful for high order meshes.")

    # }}}

    # Splitting row in eval stage
    kernel = lp.split_iname(kernel, "irow_eval", T_e_r, outer_iname="irowtile_eval")

    # Splitting column in eval stage
    for i, (T_e_c, gather_iname) in enumerate(zip(T_e_cs, trialDoF_gather_inames)):
        kernel = lp.rename_iname(
            kernel, gather_iname, "icol%d" % i, existing_ok=True
        )
        kernel = lp.split_iname(
            kernel, "icol%d" % i, T_e_c, outer_iname="icoltile%d" % i
        )

    # Splitting row in the quadr stage
    kernel = lp.split_iname(
        kernel, "irow_quadr", T_q_r, outer_iname="irowtile_quadr"
    )
    # Splitting column in quadr stage
    kernel = lp.split_iname(
        kernel, "icol%d" % n_trial, T_q_c, outer_iname="icoltile%d" % n_trial
    )

    # {{{ Also, limit the gathering of the trialDoF to the current column tile.

    for i, mv_stage, T_e_c, gather_iname in zip(
        range(n_trial), matvec_stage_descrs, T_e_cs, trialDoF_gather_inames
    ):
        for trialDoF in mv_stage.dof_names:
            kernel = lp.split_array_axis(kernel, trialDoF, 0, T_e_c)
            kernel = remove_axis(kernel, trialDoF, 0)

        kernel = lp.add_inames_to_insn(
            kernel,
            "iquad_tile,irowtile_eval",
            " or ".join("writes:%s" % trialDoF for trialDoF in mv_stage.dof_names),
        )

        if i > 1:
            # enforce a dependency of gather for the DoFs used in i+1 matvec
            # stage on the previous matvec. (helps in enforcing separate live
            # ranges).
            kernel = lp.add_dependency(
                kernel, "iname:%s_inner" % gather_iname, "tag:matvec%d" % (i - 1)
            )

    # }}}

    # {{{ Prefetch trialDoFs (not implemented)

    if tiling_config.load_input_to_shared:
        raise NotImplementedError("More like NotYetImplementedError.")

    # }}}

    # {{{ Prefetch deriv matrices

    total_shared_vars = []

    if tiling_config.load_mats_to_shared:
        from loopy.transform.data import add_prefetch_for_single_kernel

        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()

        for istage, mv_stg_descr in enumerate(matvec_stage_descrs):
            if istage < n_trial:
                # eval stage
                fetch_outer_inames = (
                    "iquad_tile,iblock,icoltile{0},irowtile_eval".format(istage)
                )
                sweep_inames = "irow_eval_inner, icol{0}_inner".format(istage)
                tr = T_e_r
                tc = T_e_cs[istage]
            else:
                # quadr stage
                fetch_outer_inames = (
                    "iquad_tile,iblock,icoltile{0},irowtile_quadr".format(istage)
                )
                sweep_inames = "irow_quadr_inner, icol{0}_inner".format(istage)
                tr = T_q_r
                tc = T_q_c

            # sweep the row, column of the tile.
            prefetch_inames = [vng("iprftch") for _ in range(2)]

            # prefetch all the derivative matrices in the current matvec stage
            for i_op_pos, prftch_from in enumerate(mv_stg_descr.deriv_matrices):
                prftch_into = vng("matvec%d_cnst_mtrix_prftch" % istage)
                total_shared_vars.append(prftch_into)

                kernel = add_prefetch_for_single_kernel(
                    kernel,
                    callables_table,
                    var_name=prftch_from,
                    sweep_inames=sweep_inames,
                    temporary_address_space=lp.AddressSpace.LOCAL,
                    dim_arg_names=prefetch_inames,
                    temporary_name=prftch_into,
                    compute_insn_id=ing("prftch_matvec%d" % istage),
                    fetch_outer_inames=fetch_outer_inames,
                    default_tag=None,
                    within="tag:matvec%d" % istage,
                )

                new_temps = kernel.temporary_variables.copy()

                lx, ly = kernel.temporary_variables[prftch_into].shape
                assert lx * ly == tr * tc
                # prefetch the matrices into a single shared memory location
                # with the appropriate offsets
                new_temps[prftch_into] = kernel.temporary_variables[
                    prftch_into
                ].copy(
                    base_storage="prftch_matrix_base",
                    offset=i_op_pos * tr * tc,
                    shape=((i_op_pos + 1) * lx, ly),
                )

                kernel = kernel.copy(temporary_variables=new_temps)

            # add dependency of the matvec stage on its prefetch instructions
            kernel = lp.add_dependency(
                kernel,
                "tag:matvec%d and (tag:eval_redn or tag:quadr_redn)" % istage,
                "id:prftch_matvec%d*" % istage,
            )
            kernel = lp.add_nosync(
                kernel,
                source="id:prftch_matvec%d*" % istage,
                sink="id:prftch_matvec%d*" % istage,
                scope="local",
                empty_ok=True,
                force=True,
            )

            # join inames to promote more coalesced memory accesses in the
            # prefetches
            kernel = lp.join_inames(
                kernel, prefetch_inames, new_iname="i_matvec%d_prftch" % istage
            )
            kernel = lp.split_iname(
                kernel, "i_matvec%d_prftch" % istage, nc * nt
            )  # , outer_tag="ilp")
            kernel = lp.split_iname(
                kernel,
                "i_matvec%d_prftch_inner" % istage,
                nt,
                inner_tag="l.0",
                outer_tag="l.1",
            )

        # {{{ prefetch of (i+1)-th matvec stage should depend on prefetch of
        # (i)th matvec stage

        for i in range(n_trial):
            kernel = lp.add_dependency(
                kernel, "id:prftch_matvec%d*" % (i + 1), "tag:matvec%d" % i
            )

        # }}}

    # }}}

    # {{{ Prefetch: Quad Weights

    if tiling_config.load_quad_weights_to_shared:
        # FIXME: instead of prefetching this we should precompute the constant
        # term which we made as a substitution.
        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()
        quad_weight_prefetch_insn = ing("quad_wt_prftch_insn")
        quad_weight_prefetch_iname = vng("iprtftch")

        if tiling_config.tiled_prefetch_of_quad_weights:
            raise NotImplementedError("Not sure if this is any fruitful!")
        else:
            sweep_inames = ["irowtile_eval", "irow_eval_inner"]
            fetch_outer_inames = "iquad_tile,iblock"

        from loopy.transform.data import add_prefetch_for_single_kernel

        kernel = add_prefetch_for_single_kernel(
            kernel,
            callables_table,
            var_name=quad_weights,
            sweep_inames=sweep_inames,
            temporary_address_space=lp.AddressSpace.LOCAL,
            dim_arg_names=(quad_weight_prefetch_iname,),
            temporary_name="cnst_quad_weight_prftch",
            compute_insn_id=quad_weight_prefetch_insn,
            fetch_outer_inames=fetch_outer_inames,
            default_tag=None,
        )

        kernel = lp.add_dependency(
            kernel,
            "tag:matvec0 and tag:eval_init",
            f"id:{quad_weight_prefetch_insn}",
        )

        kernel = lp.split_iname(
            kernel, quad_weight_prefetch_iname, nc * nt
        )  # , outer_tag="ilp")
        kernel = lp.split_iname(
            kernel,
            quad_weight_prefetch_iname + "_inner",
            nt,
            outer_tag="l.1",
            inner_tag="l.0",
        )

    # }}}

    # {{{ divide the matvec of each cell across threads

    kernel = lp.split_iname(kernel, "irow_eval_inner", nt)
    kernel = lp.split_iname(kernel, "irow_quadr_inner", nt)

    # }}}

    # {{{ privatizing the reduction accumulators

    # {{{ eval stage

    # first (ntrial-1) matvecs:
    for i in range(n_trial):
        redn_accumulators = [
            insn.assignee_name
            for insn in kernel.instructions
            if lp.match.And(
                (lp.match.Tagged("eval_init"), lp.match.Tagged("matvec%d" % i))
            )(kernel, insn)
        ]

        # privatize temporaries for logic preservation
        kernel = lp.privatize_temporaries_with_inames(
            kernel, "irow_eval_inner_outer", only_var_names=redn_accumulators
        )

        # renaming inames to decouple matvec stages
        kernel = lp.rename_iname(
            kernel,
            "irow_eval_inner_inner",
            "irow%d_inner_inner" % i,
            within="tag:matvec%d" % i,
        )
        kernel = lp.rename_iname(
            kernel,
            "irow_eval_inner_outer",
            "irow%d_inner_outer" % i,
            within="tag:matvec%d" % i,
        )

        # schedulability constraint requires irow_inner_outer to be duplicated
        # within the eval_init stage
        kernel = lp.duplicate_inames(
            kernel,
            "irow%d_inner_outer" % i,
            new_inames="irow%d_inner_outer_init" % i,
            within="tag:eval_init",
        )

        kernel = lp.tag_inames(
            kernel, "irow%d_inner_outer:unr,irow%d_inner_outer_init:unr" % (i, i)
        )
        for trialDoF in matvec_stage_descrs[i].dof_names:
            kernel = remove_axis(kernel, trialDoF, 0)

    # eval wrap up:
    kernel = lp.rename_iname(
        kernel,
        "irow_eval_inner_inner",
        "irow_eval_wrap_up_inner_inner",
        within="tag:eval_wrap_up",
    )
    kernel = lp.rename_iname(
        kernel,
        "irow_eval_inner_outer",
        "irow_eval_wrap_up_inner_outer",
        within="tag:eval_wrap_up",
    )
    kernel = lp.tag_inames(kernel, "irow_eval_wrap_up_inner_outer:unr")

    # }}}

    # {{{ quadr stage:

    redn_accumulators = [
        insn.assignee_name
        for insn in kernel.instructions
        if lp.match.Tagged("quadr_init")(kernel, insn)
    ]

    kernel = lp.privatize_temporaries_with_inames(
        kernel, "irow_quadr_inner_outer", only_var_names=redn_accumulators
    )
    kernel = lp.rename_iname(
        kernel,
        "irow_quadr_inner_inner",
        "irow%d_inner_inner" % n_trial,
        within="tag:matvec%d" % n_trial,
    )
    kernel = lp.rename_iname(
        kernel,
        "irow_quadr_inner_outer",
        "irow%d_inner_outer" % n_trial,
        within="tag:matvec%d" % n_trial,
    )

    kernel = lp.rename_iname(
        kernel,
        "irow_quadr_inner_inner",
        "irow_quadr_wrap_up_inner_inner",
        within="tag:quadr_wrap_up",
    )
    kernel = lp.rename_iname(
        kernel,
        "irow_quadr_inner_outer",
        "irow_quadr_wrap_up_inner_outer",
        within="tag:quadr_wrap_up",
    )
    kernel = lp.duplicate_inames(
        kernel,
        "irow%d_inner_outer" % n_trial,
        new_inames="irow%d_inner_outer_init" % n_trial,
        within="tag:quadr_init",
    )
    kernel = lp.tag_inames(
        kernel,
        {
            f"irow{n_trial}_inner_outer": "unr",
            f"irow{n_trial}_inner_outer_init": "unr",
            "irow_quadr_wrap_up_inner_outer": "unr",
        },
    )

    kernel = lp.add_inames_to_insn(
        kernel, "iquad_tile", "tag:quadr_init or tag:quadr_wrap_up"
    )

    # }}}

    # }}}

    kernel = lp.tag_inames(kernel, "icell:l.1, iblock:g.0")

    # {{{ tagging inames

    for i in range(n_trial + 1):
        kernel = lp.tag_inames(kernel, "irow%d_inner_inner:l.0" % i)

    kernel = lp.tag_inames(
        kernel, "irow_eval_wrap_up_inner_inner:l.0", ignore_nonexistent=True
    )
    kernel = lp.tag_inames(
        kernel, "irow_quadr_wrap_up_inner_inner:l.0", ignore_nonexistent=True
    )

    # }}}

    # {{{ setting loop priorities

    # disregard all previous priorities
    kernel = kernel.copy(loop_priority=frozenset())

    # unroll loops must be innermost
    for i in range(n_trial + 1):
        kernel = lp.prioritize_loops(
            kernel, "icol{0}_inner,irow{0}_inner_outer".format(i)
        )
    # }}}

    kernel = lp.remove_unused_inames(kernel)

    return t_unit.with_kernel(kernel), args_to_make_global


# }}}


# {{{ auto tile

WARP_SIZE = 32


@dc.dataclass(frozen=True)
class ParametricTilingCandidateGenerator:
    """
    Helper class to tune the :class:`pyop2.gpu.tile.ParametricTiling` for
    :func:`pyop2.gpu.tile.tiled_transform`. Tuning heuristic applied as
    specified in Paper xx. All the mathematical symbols used in the docs of the
    member methods are defined the paper.

    :attr fem_program: An instance of :class:`loopy.program.Program` which is
        the FEM computational kernel to be tuned.
    :attr num_param_tiling_candidates: An instance of :class:`int` denoting the
        number of parametric tiling candidates to be generate.

    See the entrypoint :meth:`__call__`
    """

    fem_program: lp.TranslationUnit
    num_param_tiling_candidates: int

    @cached_property
    def metadata(self):
        knl = self.fem_program.default_entrypoint
        return inference_which_should_ideally_be_done_by_passing_metadata(knl)[1]

    @cached_property
    def nquad(self):
        return self.metadata.n_quad

    @cached_property
    def matvec_stages(self):
        return self.metadata.matvec_stage_descrs

    @cached_property
    def n_trial_stages(self):
        return self.metadata.n_trial_stages

    @cached_property
    def n_eval_terms(self):
        return len(self.metadata.eval_results)

    @cached_property
    def n_trialDoFs(self):
        return self.metadata.n_trialDoFs

    @cached_property
    def n_outDoF(self):
        return self.metadata.n_outDoF

    @cached_property
    def n_trial_derivs(self):
        return self.metadata.n_trial_derivs

    @cached_property
    def trialDoF_shapes(self):
        sizes = [
            [
                (
                    self.fem_program.default_entrypoint.temporary_variables[
                        dof_name
                    ].shape
                )
                for dof_name in mv_stage.dof_names
            ]
            for mv_stage in self.matvec_stages[:-1]
        ]
        return sizes

    @cached_property
    def outDoF_shape(self):
        outDoF = self.metadata.outDoF
        return self.fem_program.default_entrypoint.temporary_variables[outDoF].shape

    @cached_property
    def coords_shape(self):
        coords = self.metadata.coords
        return self.fem_program.default_entrypoint.temporary_variables[coords].shape

    @cached_property
    def deriv_mat_shapes(self):
        sizes = [
            [
                (
                    self.fem_program.default_entrypoint.temporary_variables[
                        mat_name
                    ].shape
                )
                for mat_name in mv_stage.deriv_matrices
            ]
            for mv_stage in self.matvec_stages
        ]
        return sizes

    def get_nsync(self, tiling_config):
        """
        Returns the number of block level synchronization instructions in a
        single kernel execution.
        """
        tiles = tiling_config.operator_tile_descriptions
        T_e_r = tiles[0][0]
        T_e_cs = [tile[1] for tile in tiles[:-1]]
        T_q_r = tiles[-1][0]
        T_q_c = tiles[-1][1]
        (quad_tile_len,) = tiling_config.quad_rowtile_lengths

        def get_nsync_for_quad(nq):
            return (ceil(self.n_outDoF / T_q_r)) * (ceil(nq / T_q_c)) + sum(
                ceil(nq / T_e_r) * ceil(n_trialDoF / T_e_c)
                for n_trialDoF, T_e_c in zip(self.n_trialDoFs, T_e_cs)
            )

        return floor(self.nquad / quad_tile_len) * get_nsync_for_quad(
            quad_tile_len
        ) + get_nsync_for_quad(self.nquad % quad_tile_len)

    def get_shared_mem_allocated(self, tiling_config):
        """
        Returns the shared memory usage for *tiling_config* in bytes.
        """
        nc = tiling_config.ncells_per_block
        tiles = tiling_config.operator_tile_descriptions
        (quad_tile_len,) = tiling_config.quad_rowtile_lengths
        n_eval_mats = [
            len(mv_stage.deriv_matrices) for mv_stage in self.matvec_stages[:-1]
        ]
        n_q_mats = len(self.matvec_stages[-1].deriv_matrices)

        shared_mem = (
            max(
                n_mat * tile[0] * tile[1]
                for n_mat, tile in zip(
                    n_eval_mats
                    + [
                        n_q_mats,
                    ],
                    tiles,
                )
            )
            + quad_tile_len
            + nc * quad_tile_len * self.n_eval_terms
        )

        return shared_mem * 8

    def get_eta_simd(self, tiling_config):
        nc = tiling_config.ncells_per_block
        nwi = tiling_config.nthreads_per_cell
        return (nc * nwi) / (32 * ceil(nc * nwi / 32))

    def get_eta_load_balance(self, tiling_config):
        tiles = tiling_config.operator_tile_descriptions
        nwi = tiling_config.nthreads_per_cell
        (quad_tile_len,) = tiling_config.quad_rowtile_lengths
        T_e_r = tiles[0][0]
        T_q_r = tiles[-1][0]

        def get_flops_executed_for_nq(nq):
            n1 = floor(nq / T_e_r)
            n2 = floor(self.n_outDoF / T_q_r)
            n3 = (
                nwi
                * ceil(T_e_r / nwi)
                * sum(
                    n_deriv * n_dof
                    for n_deriv, n_dof in zip(self.n_trial_derivs, self.n_trialDoFs)
                )
            )
            n4 = nwi * ceil(T_q_r / nwi) * self.n_eval_terms * nq
            n5 = (
                nwi
                * ceil((nq % T_e_r) / nwi)
                * sum(
                    n_deriv * n_dof
                    for n_deriv, n_dof in zip(self.n_trial_derivs, self.n_trialDoFs)
                )
            )
            n6 = nwi * ceil((self.n_outDoF % T_q_r) / nwi) * self.n_eval_terms * nq

            return n3 * n1 + n5 + n4 * n2 + n6

        flops_executed = floor(
            self.nquad / quad_tile_len
        ) * get_flops_executed_for_nq(quad_tile_len) + get_flops_executed_for_nq(
            self.nquad % quad_tile_len
        )

        useful_flops = (
            self.nquad
            * sum(
                n_deriv * n_dof
                for n_deriv, n_dof in zip(self.n_trial_derivs, self.n_trialDoFs)
            )
            + self.n_eval_terms * self.nquad * self.n_outDoF
        )

        eta_load = useful_flops / flops_executed

        return eta_load

    def get_theoretical_blocks_per_sm(self, tiling_config):
        """
        Returns the number of blocks residing on a Streaming Multiprocessor.
        """
        S = self.get_shared_mem_allocated(tiling_config)
        dev = cuda.Context.get_device()
        Smax_per_sm = dev.get_attribute(
            cuda.device_attribute.MAX_SHARED_MEMORY_PER_MULTIPROCESSOR
        )
        Smax_per_block = dev.get_attribute(
            cuda.device_attribute.MAX_SHARED_MEMORY_PER_BLOCK
        )
        Wmax = 16
        blocks_per_sm = min(Smax_per_sm // S if S < Smax_per_block else 0, Wmax)
        return blocks_per_sm

    def get_theoretical_warps_per_sm(self, tiling_config):
        """
        Returns the number of warps residing on a Streaming Multiprocessor.
        """
        blocks_per_sm = self.get_theoretical_blocks_per_sm(tiling_config)
        warps_per_block = ceil(
            tiling_config.nthreads_per_cell * tiling_config.ncells_per_block / 32
        )
        warps_per_sm = min(blocks_per_sm * warps_per_block, 16)
        return warps_per_sm

    def get_effective_warps_per_sm(self, tiling_config):
        """
        Returns the effective number of warps residing on a Streaming Multiprocessor.
        """
        return (
            self.get_eta_load_balance(tiling_config)
            * self.get_eta_simd(tiling_config)
            * self.get_theoretical_warps_per_sm(tiling_config)
        )

    def get_effective_blocks_per_sm(self, tiling_config):
        """
        Returns the effective number of warps residing on a Streaming Multiprocessor.
        """
        return (
            self.get_eta_load_balance(tiling_config)
            * self.get_eta_simd(tiling_config)
            * self.get_theoretical_blocks_per_sm(tiling_config)
        )

    @memoize_method
    def estimated_exec_time(self, tiling_config):
        """
        Returns a metric proportional to the execution time for a
        configuration.
        """
        T_e_r = tiling_config.operator_tile_descriptions[0][0]
        (quad_tile_len,) = tiling_config.quad_rowtile_lengths
        nwi = tiling_config.nthreads_per_cell
        nwarps = self.get_effective_warps_per_sm(tiling_config)
        nblocks = self.get_effective_blocks_per_sm(tiling_config)
        nsync = self.get_nsync(tiling_config)
        effective_global_bw = 21 if nwarps > 8 else 20 * (nwarps / 8)
        effective_shared_bw = (
            min(1100, 900 + 30 * (nwarps - 10))
            if nwarps > 10
            else 900 * (nwarps / 10)
        )

        # gather phase times
        num_times_trial_dofs_gather_per_quad_tile = lambda qt: ceil(  # noqa: E731
            qt / T_e_r
        )
        num_times_trial_dofs_gathered = floor(
            self.nquad / quad_tile_len
        ) * num_times_trial_dofs_gather_per_quad_tile(
            quad_tile_len
        ) + num_times_trial_dofs_gather_per_quad_tile(
            self.nquad % quad_tile_len
        )

        gather_phase_gbytes = 8e-9 * (
            num_times_trial_dofs_gathered
            * sum(
                sum(np.prod(dof_shape) for dof_shape in mv_stage_dof_shapes)
                for mv_stage_dof_shapes in self.trialDoF_shapes
            )
            + np.prod(self.coords_shape)
        )
        gather_phase_time = gather_phase_gbytes / effective_global_bw

        # scatter phase times
        scatter_phase_gbytes = (
            8e-9 * (np.prod(self.outDoF_shape)) * (ceil(self.nquad / quad_tile_len))
        )
        scatter_phase_time = scatter_phase_gbytes / effective_global_bw

        # reading in the data for quad weights/deriv matrices
        read_constant_data_into_smem_gbytes = (
            8e-9
            * (
                sum(
                    sum(
                        np.prod(deriv_mat_shape)
                        for deriv_mat_shape in mv_stage_deriv_mat_shapes
                    )
                    for mv_stage_deriv_mat_shapes in self.deriv_mat_shapes
                )
                + self.nquad
            )
            / tiling_config.ncells_per_block
        )

        read_constants_data_into_smem_time = (
            read_constant_data_into_smem_gbytes / effective_global_bw
        )

        # eval phase times
        eval_phase_smem_read_gbytes = 8e-9 * sum(
            n_trial_deriv * np.prod(mv_stage_deriv_mat_shapes[0])
            for n_trial_deriv, mv_stage_deriv_mat_shapes in zip(
                self.n_trial_derivs, self.deriv_mat_shapes
            )
        )
        eval_phase_smem_read_time = eval_phase_smem_read_gbytes / effective_shared_bw

        # quadr phase times
        quadr_phase_mat_smem_read_gbytes = 8e-9 * (
            self.n_eval_terms * np.prod(self.deriv_mat_shapes[-1][0])
        )
        quadr_phase_rhs_smem_read_gbytes = 8e-9 * (
            self.n_eval_terms * np.prod(self.deriv_mat_shapes[-1][0])
        )
        quadr_phase_smem_read_time = (
            quadr_phase_mat_smem_read_gbytes / effective_shared_bw
            + quadr_phase_rhs_smem_read_gbytes / effective_shared_bw
        )

        total_time = (
            gather_phase_time
            + scatter_phase_time
            + read_constants_data_into_smem_time
            + eval_phase_smem_read_time
            + quadr_phase_smem_read_time
        )
        return (total_time, nsync)

        return 4.0 / (nwarps) + nsync / nblocks + nwi / 8

    def __call__(self) -> Tuple[ParametricTiling, ...]:
        from itertools import product

        threads_to_cells = {}

        def eta_simd(nc, nt):
            return (nc * nt) / (32.0 * ceil(nc * nt / 32))

        def get_eta_shared_mem_alias(tiles):
            nmats = [len(mv_stage.deriv_matrices) for mv_stage in self.matvec_stages]
            min_sm_usage_in_a_stage = min(
                nmat * tr * tc for nmat, (tr, tc) in zip(nmats, tiles)
            )
            max_sm_usage_in_a_stage = max(
                nmat * tr * tc for nmat, (tr, tc) in zip(nmats, tiles)
            )

            return min_sm_usage_in_a_stage / max_sm_usage_in_a_stage

        for nc in range(1, 70):
            for nt in range(1, 20):
                if eta_simd(nc, nt) > 0.97 and (nc * nt <= 256):
                    if nt in threads_to_cells:
                        threads_to_cells[nt].append(nc)
                    else:
                        threads_to_cells[nt] = [nc]

        tiles = []

        for nquad_tiles in range(1, 14):
            quad_tile_len = ceil(self.nquad / nquad_tiles)
            if nquad_tiles > 1 and quad_tile_len == ceil(
                self.nquad / (nquad_tiles - 1)
            ):
                continue

            for i in range(1, ceil(sqrt(quad_tile_len) + 1)):
                T_e_r = ceil(quad_tile_len / i)
                for j in product(
                    *[
                        range(1, ceil(sqrt(ntrialDoF)) + 1)
                        for ntrialDoF in self.n_trialDoFs
                    ]
                ):
                    T_e_cs = tuple(
                        ceil(ntrialDoF / jj)
                        for ntrialDoF, jj in zip(self.n_trialDoFs, j)
                    )
                    for k in range(1, ceil(sqrt(self.n_outDoF)) + 1):
                        T_q_r = ceil(self.n_outDoF / k)
                        for ell in range(1, ceil(sqrt(quad_tile_len)) + 1):
                            T_q_c = ceil(quad_tile_len / ell)
                            current_tile = tuple(
                                (T_e_r, T_e_c) for T_e_c in T_e_cs
                            ) + ((T_q_r, T_q_c),)
                            if get_eta_shared_mem_alias(current_tile) >= 0.8:
                                tiles.append((quad_tile_len, current_tile))

        params = []

        for quad_tile_len, tile in tiles:
            for threads in threads_to_cells:
                for cells in threads_to_cells[threads]:
                    params.append(
                        ParametricTiling(
                            cells,
                            threads,
                            tile,
                            (quad_tile_len,),
                            False,
                            False,
                            True,
                            True,
                            False,
                            False,
                        )
                    )

        # sort the parameters with highest occupancy.
        params.sort(key=lambda P: self.estimated_exec_time(P))

        return tuple(params[: self.num_param_tiling_candidates])


# }}}


@lp.memoize_on_disk
def get_transform_candidates(
    fem_kernel: lp.TranslationUnit,
) -> Tuple[TransformCandidate, ...]:
    return ParametricTilingCandidateGenerator(fem_kernel, 10)() + (SWIPC(),)


def _transform_kernel_with_candidate(
    kernel: lp.TranslationUnit, candidate: TransformCandidate
) -> Tuple[lp.TranslationUnit, Tuple[np.ndarray, ...]]:
    if isinstance(candidate, SWIPC):
        from pyop2.transforms.snpt import split_n_across_workgroups

        return split_n_across_workgroups(kernel, 32)
    elif isinstance(candidate, ParametricTiling):
        return tiled_transform(kernel, candidate)
    else:
        raise NotImplementedError(type(candidate))


def _np_ary_to_cuda_mem(ary: np.ndarray) -> cuda.DeviceAllocation:
    assert isinstance(ary, np.ndarray)
    ary_gpu = cuda.mem_alloc(ary.nbytes)
    cuda.memcpy_htod(src=ary, dest=ary_gpu)
    return ary_gpu


def get_empirically_best_candidate(
    t_unit: lp.TranslationUnit,
    *,
    args: Sequence[Union[int, cuda.DeviceAllocation]],
    candidates: Tuple[TransformCandidate, ...],
) -> TransformCandidate:

    best_time = np.inf
    best_candidate = None

    nminrounds = 15
    nwarmup = 5
    mintime = 0.1

    copied_args: List[Union[int, cuda.DeviceAllocation]] = []
    epoint_knl = t_unit.default_entrypoint
    for arg_gpu, lpy_arg in zip(args, epoint_knl.args, strict=True):
        if lpy_arg.name in epoint_knl.get_written_variables():
            assert isinstance(arg_gpu, cuda_np.GPUArray)
            arg_gpu = cuda_np.empty_like(arg_gpu)

        copied_args.append(arg_gpu)

    for candidate in candidates:

        transformed_t_unit, extra_args = _transform_kernel_with_candidate(
            t_unit, candidate
        )
        print(transformed_t_unit)
        1 / 0
        assert all(isinstance(extra_arg, np.ndarray) for extra_arg in extra_args)

        code = lp.generate_code_v2(transformed_t_unit).device_code()

        glens, llens = transformed_t_unit.get_grid_size_upper_bounds_as_exprs()
        from pymbolic import evaluate

        grid = tuple(
            (
                int(evaluate(glens[i], {"start": args[0], "end": args[1]}))
                if i < len(glens)
                else 1
            )
            for i in range(2)
        )
        block = tuple(
            (
                int(evaluate(llens[i], {"start": args[0], "end": args[1]}))
                if i < len(llens)
                else 1
            )
            for i in range(3)
        )

        executable_knl = SourceModule(
            code, options=["-use_fast_math", "-w"]
        ).get_function(t_unit.default_entrypoint.name)
        executable_knl.prepare("i" * 2 + "P" * len(args[2:]) + "P" * len(extra_args))
        extra_args = tuple(
            _np_ary_to_cuda_mem(extra_arg) for extra_arg in extra_args
        )

        for i in range(nwarmup):
            executable_knl.prepared_call(grid, block, *copied_args, *extra_args)

        runtimes = []

        # execute the kernel for a minimum of "nminrounds" of non-warmup
        # runs and such that it run for at least 0.1s
        while (len(runtimes) < nminrounds) or sum(runtimes) < mintime:
            start_evt = cuda.Event()
            end_evt = cuda.Event()
            start_evt.record()

            for i in range(10):
                executable_knl.prepared_call(grid, block, *copied_args, *extra_args)

            end_evt.record()
            end_evt.synchronize()
            runtimes.append(1e-3 * (end_evt.time_since(start_evt) / 10))

        candidate_runtime = np.median(runtimes)

        if candidate_runtime < best_time:
            best_time = candidate_runtime
            best_candidate = candidate

    return best_candidate


def _preprocess_tunit_for_autotiling(
    t_unit: lp.TranslationUnit,
) -> lp.TranslationUnit:
    kernel = t_unit.default_entrypoint

    # remove noops
    noop_insns = set(
        [
            insn.id
            for insn in kernel.instructions
            if isinstance(insn, (lp.NoOpInstruction, lp.CInstruction))
        ]
    )
    kernel = lp.remove_instructions(kernel, noop_insns)
    kernel = remove_unnecessary_deps(kernel)
    kernel = lp.simplify_indices(kernel)

    return t_unit.with_kernel(kernel)


def autotuned_tiling(
    t_unit, arguments: Tuple[Union[int, "cuda.DeviceAllocation"], ...]
):
    """
    Returns ``(transformed_kernel, args_to_make_global)``, where
    ``transformed_kernel`` is the kernel transformed via the auto-tuning
    approach outlined in ``PAPER-link (TODO)`` and ``args_to_make_global``
    is an instance of ``pycuda.Array`` which come in as additional variables
    to the kernel which were introduced during the kernel transform process.

    :arg kernel: The FEM-action kernel which is to be transformed.
    :arg arguments: The arguments with which the *kernel* is invoked.
        The transformation pathway works by auto-tuning which needs a dummy
        set of arguments to operate. It is important to note that the
        transformed kernel will be valid for other arguments as well.
    """
    # Step.0: Preprocessing to simply transform implementation
    t_unit = _preprocess_tunit_for_autotiling(t_unit)

    # Step.1: Get candidates (memoized)
    candidates = get_transform_candidates(t_unit)

    # Step.2: Find the best candidate
    best_candidate = get_empirically_best_candidate(
        t_unit, candidates=candidates, args=arguments
    )

    # Step. 3. Transform the kernel with the best candidate
    return _transform_kernel_with_candidate(t_unit, best_candidate)


# vim: fdm=marker
