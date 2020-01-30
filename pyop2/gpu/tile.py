import loopy as lp
import numpy as np
import pycuda.driver as cuda
from math import ceil, sqrt, floor
from pytools import memoize_method
from pycuda.compiler import SourceModule
from pyop2.utils import cached_property
from pytools import ImmutableRecord


# {{{ implementing the tiling transformation

class TilingConfiguration(ImmutableRecord):
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
        prefeteched to shared memory?
    :attr load_input_to_shared: Should the input DoFs be prefetched to shared
        memory?
    :attr load_mats_to_shared: Should the local FEM operator matrices be loaded
        to shared memory?
    :attr load_quad_weights_to_shared: Should the quadrature weigts be loaded
        to shared memory?
    :attr tiled_prefetch_of_inputs: If input DoFs are prefetched to shared
        memory, should they be prefetched in tile lengths?
    :attr tiled_prefetch_of_quad_weights: If the quadrature weights are
        prefethced to shared memory, should they in prefetched in tile lengths?
    """
    def __init__(self,
            ncells_per_block,
            nthreads_per_cell,
            operator_tile_descriptions,
            quad_rowtile_lengths,
            load_coordinates_to_shared,
            load_input_to_shared,
            load_mats_to_shared,
            load_quad_weights_to_shared,
            tiled_prefetch_of_inputs,
            tiled_prefetch_of_quad_weights):
        super(TilingConfiguration, self).__init__(
                ncells_per_block=ncells_per_block,
                nthreads_per_cell=nthreads_per_cell,
                operator_tile_descriptions=operator_tile_descriptions,
                quad_rowtile_lengths=quad_rowtile_lengths,
                load_coordinates_to_shared=load_coordinates_to_shared,
                load_input_to_shared=load_input_to_shared,
                load_mats_to_shared=load_mats_to_shared,
                load_quad_weights_to_shared=load_quad_weights_to_shared,
                tiled_prefetch_of_inputs=tiled_prefetch_of_inputs,
                tiled_prefetch_of_quad_weights=tiled_prefetch_of_quad_weights)


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
            is_output_only=not tv.read_only)
    return arg


class KernelMetadata(ImmutableRecord):
    def __init__(self, **kwargs):
        assert isinstance(kwargs['quad_iname'], str)
        assert isinstance(kwargs['inputDoFs'], list)
        assert isinstance(kwargs['coords'], str)
        assert isinstance(kwargs['outputDoF'], str)
        assert isinstance(kwargs['outputDoF_init_iname'], str)
        assert isinstance(kwargs['doF_inames_in_quad_stage'], list)
        assert isinstance(kwargs['doF_iname_in_basis_stage'], str)
        assert isinstance(kwargs['scatter_iname'], str)
        assert isinstance(kwargs['nquad'], int)
        assert isinstance(kwargs['n_inputDoFs'], list)
        assert isinstance(kwargs['n_outputDoF'], int)
        assert isinstance(kwargs["op_matrices"], frozenset)
        assert isinstance(kwargs["quad_weights"], str)
        assert isinstance(kwargs["matvec_stage_to_op_matrices"], list)

        # FIXME: non-obvious styling
        # just choose the lengthy route
        assert len(kwargs) == 14
        super(KernelMetadata, self).__init__(**kwargs)


def work_which_should_be_done_by_passing_metadata(kernel):
    from pymbolic.primitives import Variable

    # quad iname
    # Logic: assumes that there is only one iname responsible for the
    # quadrature. TPEs do not fit within this model
    quad_iname, = [iname for iname in kernel.all_inames() if iname.startswith('form_ip')]

    # inputDof_x_outputDofs_x_coords: A set containing the variable names for the
    # temporaries of inputDofs, outputDofs and the coordinates.
    inputDof_x_outputDof_x_coords = set().union(*(insn.write_dependency_names() for
            insn in kernel.instructions if 'gather' in insn.tags)) - kernel.all_inames()

    # outputDof names
    # Logic: The only temporaries which are written during basis stage tagged
    # by TSFC.
    outputDoF = set()
    for insn in kernel.instructions:
        if 'basis' in insn.tags:
            outputDoF.update(insn.write_dependency_names()-kernel.all_inames())

    outputDoF, = outputDoF

    # coords name
    # Logic: assumming that the coordinate transformation is affine i.e. one
    # Jacobian computation for each cell.
    coords = set()
    for insn in kernel.instructions:
        if 'quadrature' in insn.tags and (insn.within_inames == frozenset(["n"])):
            coords = coords | (insn.read_dependency_names() &
                    inputDof_x_outputDof_x_coords)

    coords, = coords

    # inputDof names
    inputDoFs = list(inputDof_x_outputDof_x_coords - frozenset([coords, outputDoF]))

    # 1. Figure out the input basis iname

    # {{{ scatter iname

    # Logic: Assumes that there is only DoFs of one component of the basis
    # functions computed per kernel i.e. one component in a mixed FEM setting.

    scatter_insn, = [insn for insn in kernel.instructions if 'scatter' in
            insn.tags]
    scatter_map = scatter_insn.assignee.index_tuple[0]
    scatter_iname, = set(scatter_map.index_tuple) - set([Variable('n')])
    scatter_iname = scatter_iname.name

    # }}}

    # {{{ basis init iname

    # iname over the DoFs for the input variable inputDofs[i]

    outputDoF_init_iname, = [insn.assignee.index_tuple[1].name for insn in
            kernel.instructions if ('gather' in insn.tags) and (outputDoF in
                insn.write_dependency_names())]

    # }}}

    # Assumption: only one variable for the outputDoF supported.
    (doF_iname_in_basis_stage,), = set([insn.within_inames - frozenset(['n', quad_iname]) for insn in kernel.instructions if 'basis' in insn.tags])

    doF_inames_in_quad_stage = []
    for inputDoF in inputDoFs:
        iname, = frozenset().union(*(insn.within_inames for insn in
            kernel.instructions if inputDoF in insn.read_dependency_names())) - frozenset(["n", quad_iname])
        doF_inames_in_quad_stage.append(iname)

    # {{{ tagging the stages of the kernel

    #TODO: Should be interpreted in TSFC

    new_insns = []

    done_with_gather=False
    done_with_jacobi_eval = False
    done_with_quad_init = False
    done_with_quad_reduction = False
    done_with_quad_wrap_up = False
    done_with_basis_reduction = False

    for insn in kernel.instructions:
        if not done_with_gather:
            if 'gather' not in insn.tags:
                done_with_gather = True
            else:
                new_insns.append(insn)
                continue
        if not done_with_jacobi_eval:
            if quad_iname in insn.within_inames:
                done_with_jacobi_eval = True

            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["jacobi_eval"])))
                continue
        if not done_with_quad_init:
            if frozenset(doF_inames_in_quad_stage) & insn.within_inames:
                done_with_quad_init = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_init"])))
                continue
        if not done_with_quad_reduction:
            if frozenset(doF_inames_in_quad_stage) & insn.within_inames:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_redn"])))
                continue
            else:
                done_with_quad_reduction = True
        if not done_with_quad_wrap_up:
            if 'basis' in insn.tags:
                done_with_quad_wrap_up = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_wrap_up"])))
                continue
        if not done_with_basis_reduction:
            if quad_iname not in insn.within_inames:
                done_with_basis_reduction = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["basis_redn"])))
                continue
        new_insns.append(insn)

    kernel = kernel.copy(instructions=new_insns)
    assert done_with_basis_reduction

    # }}}

    # {{{ compute nDofs, nquads

    nquad = int(lp.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds(quad_iname, constants_only=True).size))
    n_inputDoFs = [int(lp.symbolic.pw_aff_to_expr(
        kernel.get_iname_bounds(iname, constants_only=True).size)) for iname in
        doF_inames_in_quad_stage]
    n_outputDoF = int(lp.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds(outputDoF_init_iname,
                constants_only=True).size))

    # }}}

    # {{{ tag the different matvecs of a kernel

    from loopy.match import parse_match

    for i, inputDoF in enumerate(inputDoFs):
        insn_id = None
        within = parse_match("writes:{}".format(inputDoF))
        for insn in kernel.instructions:
            if within(kernel, insn):
                insn_id = insn.id
                break

        assert insn_id
        from loopy.kernel.tools import find_recursive_reverse_dependencies
        matvec_insn_ids = find_recursive_reverse_dependencies(kernel, set([insn_id]))
        kernel = lp.tag_instructions(kernel, 'matvec%d' % i,
                '(' + ' or '.join(['id:%s' % matvec_insn_id for matvec_insn_id in
                    matvec_insn_ids]) + ') and tag:quadrature')
        vars_written_in_matvec = frozenset().union(*(
            insn.write_dependency_names() for insn in kernel.instructions if
            'matvec%d' % i in insn.tags))
        quad_init_insn_ids = [insn.id for insn in kernel.instructions if
                (insn.write_dependency_names() & vars_written_in_matvec) and
                'quad_init' in insn.tags]

        kernel = lp.tag_instructions(kernel, 'matvec%d' % i,
                ' or '.join(['id:%s' % quad_init_insn_id for quad_init_insn_id in
                    quad_init_insn_ids]))

    kernel = lp.tag_instructions(kernel, 'matvec%d' % (i+1),
            'reads:{0} or writes:{0}'.format(outputDoF))
    kernel = lp.tag_instructions(kernel, 'basis_init', 'tag:gather and'
            ' tag:matvec%d' % (i+1))
    kernel = lp.tag_instructions(kernel, 'basis_wrap_up', 'tag:scatter and'
            ' tag:matvec%d' % (i+1))

    # }}}

    # {{{ identifying the constants

    op_matrices = frozenset(tv.name for tv in kernel.temporary_variables.values()
            if tv.initializer is not None and len(tv.initializer.shape) != 1)
    matvec_stage_to_op_matrices = []

    for i in range(len(inputDoFs)+1):
        within = parse_match("tag:matvec%d" % i)
        stage_matrices = frozenset().union(*(
            (insn.read_dependency_names() & op_matrices)
            for insn in kernel.instructions if within(kernel, insn)))
        matvec_stage_to_op_matrices.append(stage_matrices)

    quad_weights, = [tv.name for tv in kernel.temporary_variables.values()
            if tv.initializer is not None and len(tv.initializer.shape) == 1]

    # }}}

    return kernel, KernelMetadata(
            quad_iname=quad_iname,
            inputDoFs=inputDoFs,
            coords=coords,
            outputDoF=outputDoF,
            doF_inames_in_quad_stage=doF_inames_in_quad_stage,
            outputDoF_init_iname=outputDoF_init_iname,
            scatter_iname=scatter_iname,
            doF_iname_in_basis_stage=doF_iname_in_basis_stage,
            nquad=nquad,
            n_inputDoFs=n_inputDoFs,
            n_outputDoF=n_outputDoF,
            op_matrices=op_matrices,
            quad_weights=quad_weights,
            matvec_stage_to_op_matrices=matvec_stage_to_op_matrices
            )


def tiled_transform(kernel, callables_table, tiling_config):
    """
    :param tiling_config: An instance of :class:`pyop2.gpu.tiling_config
    """

    assert isinstance(tiling_config, TilingConfiguration)

    # {{{ remove noops

    noop_insns = set([insn.id for insn in kernel.instructions if
            isinstance(insn, lp.NoOpInstruction)])
    kernel = lp.remove_instructions(kernel, noop_insns)

    from loopy.transform.instruction import remove_unnecessary_deps
    kernel = remove_unnecessary_deps(kernel)

    # }}}

    # {{{ Inferring variables

    kernel, metadata = work_which_should_be_done_by_passing_metadata(kernel)
    quad_iname = metadata.quad_iname
    inputDoFs = metadata.inputDoFs
    coords = metadata.coords
    outputDoF = metadata.outputDoF
    doF_inames_in_quad_stage = metadata.doF_inames_in_quad_stage
    outputDoF_init_iname = metadata.outputDoF_init_iname
    scatter_iname = metadata.scatter_iname
    doF_iname_in_basis_stage = metadata.doF_iname_in_basis_stage
    nquad = metadata.nquad
    n_inputDoFs = metadata.n_inputDoFs
    n_outputDoF = metadata.n_outputDoF
    op_matrices = metadata.op_matrices
    quad_weights = metadata.quad_weights
    matvec_stage_to_op_matrices = metadata.matvec_stage_to_op_matrices

    # }}}

    nc = tiling_config.ncells_per_block
    nt = tiling_config.nthreads_per_cell
    op_tile_descrs = tiling_config.operator_tile_descriptions
    q_tile_lens = tiling_config.quad_rowtile_lengths[:]

    if op_tile_descrs == ():
        op_tile_descrs = tuple((nquad, nDoF) for nDoF in n_inputDoFs) + ((n_outputDoF, nquad),)

    if q_tile_lens == ():
        q_tile_lens = (nquad,)
        # if later firedrake decides to fuse the kernels then we might have
        # multiple quadrature loops per kernel and that's the reason we are
        # allowing a tuple input. One spec for each quadrature loop.

    assert len(op_tile_descrs) == len(inputDoFs) + 1  # currently we have only outputDoF
    assert len(q_tile_lens) == 1  # currently we have only one outputDoF

    if q_tile_lens != (nquad,):
        raise NotImplementedError("Yep, not implemented!")

    assert all(len(tile_descr) == 2 for tile_descr in op_tile_descrs)

    # {{{ privatize temps for function evals and make them LOCAL

    #FIXME: Need these variables from TSFC's metadata
    evaluation_variables = (set().union(*[insn.write_dependency_names() for insn in kernel.instructions if 'quad_wrap_up' in insn.tags])
            & set().union(*[insn.read_dependency_names() for insn in kernel.instructions if 'basis' in insn.tags]))

    kernel = lp.privatize_temporaries_with_inames(kernel, quad_iname,
            evaluation_variables)
    new_temps = kernel.temporary_variables.copy()
    for eval_var in evaluation_variables:
        new_temps[eval_var] = new_temps[eval_var].copy(
                address_space=lp.AddressSpace.LOCAL)
    kernel = kernel.copy(temporary_variables=new_temps)

    # }}}

    # {{{ cast scalars which occur in both the matvecs as substs

    from loopy.match import parse_match

    within = parse_match('tag:quad_wrap_up and not tag:matvec*')
    proxy_quad_wt_scalars = [insn.assignee.name for insn in kernel.instructions if
            within(kernel, insn)]

    for proxy_quad_wt_scalar in proxy_quad_wt_scalars:
        assert kernel.temporary_variables[proxy_quad_wt_scalar].shape == ()
        kernel = lp.assignment_to_subst(kernel, proxy_quad_wt_scalar,
                extra_arguments=(quad_iname,))

    assert len(kernel.substitutions) == len(proxy_quad_wt_scalars)
    kernel = lp.expand_subst(kernel)

    # }}}

    #{{{ Duplicate inames to separate transformation logic for different matvecs

    for i, iname in enumerate(doF_inames_in_quad_stage):
        kernel = lp.duplicate_inames(kernel, quad_iname, "tag:matvec%d" % i,
            "irow%d" % i)
        kernel = lp.duplicate_inames(kernel, iname, "tag:matvec%d" % i,
            "icol%d" % i)

    kernel = lp.duplicate_inames(kernel, quad_iname, "tag:matvec%d" % (i+1),
        "icol%d" % (i+1))
    kernel = lp.duplicate_inames(kernel, doF_iname_in_basis_stage, "tag:matvec%d" % (i+1),
        "irow%d" % (i+1))

    # }}}

    # {{{ change address space of constants to '__global'

    old_temps = kernel.temporary_variables.copy()
    args_to_make_global = [tv.initializer.flatten() for tv in old_temps.values() if tv.initializer is not None]

    new_temps = dict((tv.name, tv) for tv in old_temps.values() if tv.initializer is None)
    kernel = kernel.copy(
            args=kernel.args+[_make_tv_array_arg(tv) for tv in old_temps.values() if tv.initializer is not None],
            temporary_variables=new_temps)

    # }}}

    from loopy.loop import fuse_loop_domains
    kernel = fuse_loop_domains(kernel)

    from loopy.transform.data import remove_unused_axes_in_temporaries
    kernel = remove_unused_axes_in_temporaries(kernel)

    # Realize CUDA blocks
    kernel = lp.split_iname(kernel, "n", nc,
            outer_iname="iblock", inner_iname="icell")

    kernel = lp.privatize_temporaries_with_inames(kernel, 'icell',
            only_var_names=evaluation_variables)

    # cut down the size of the number of basis coeffs written by each
    # thread(if there are multiple threads)
    kernel = lp.rename_iname(kernel, scatter_iname,
            'irow%d' % len(inputDoFs), True)
    kernel = lp.rename_iname(kernel, outputDoF_init_iname,
            'irow%d' % len(inputDoFs), True)

    from loopy.transform.make_scalar import remove_axis
    kernel = remove_axis(kernel, outputDoF, 0)

    kernel = lp.add_dependency(kernel, 'tag:quad_init and tag:matvec0', 'tag:jacobi_eval')

    for i, _ in enumerate(inputDoFs):
        kernel = lp.add_dependency(kernel,
                '(tag:quad_init or tag:basis_init) and tag:matvec%d' % (i+1),
                'tag:quad_wrap_up and tag:matvec%d' % i)

    if tiling_config.load_coordinates_to_shared:
        # FIXME: This configuration parameter seems unnecessary as of now. I
        # might choose not to support it.
        kernel = lp.privatize_temporaries_with_inames(kernel, 'icell',
                [coords])
        kernel = lp.assignment_to_subst(kernel, coords)
        raise NotImplementedError("This might be only useful for high order"
                " meshes.")

    # Splitting for tiles in matvec1
    # Realizing tiles in the quadrature stage

    for i, (t_r, t_c) in enumerate(op_tile_descrs):
        kernel = lp.split_iname(kernel, "irow%d" % i, t_r,
                outer_iname='irowtile%d' % i)
        kernel = lp.split_iname(kernel, "icol%d" % i, t_c,
                outer_iname='icoltile%d' % i)


    # {{{ Prefetch inputDoFs

    if tiling_config.load_input_to_shared:
        raise NotImplementedError("More like NotYetImplementedError.")
        kernel = lp.privatize_temporaries_with_inames(kernel, 'icell',
                only_var_names=inputDoFs)
        from loopy.transform.precompute import precompute_for_single_kernel
        for i, inputDoF in enumerate(inputDoFs):
            kernel = lp.assignment_to_subst(kernel, inputDoF)
            input_prcmpt_iname = 'input_basis_prcmpt'
            if tiling_config.tiled_prefetch_of_inputs:
                sweep_inames = (doF_inames_in_quad_stage[i]+'_inner', 'icell')
                outer_inames = 'iblock,icoltile_matvec1,irowtile_matvec1'
            else:
                sweep_inames = ('icoltile_matvec1', doF_inames_in_quad_stage[i]+'_inner', 'icell')
                outer_inames = 'iblock'
            kernel = precompute_for_single_kernel(kernel, callables_table,
                    subst_use=doF_inames_in_quad_stage[i]+'_subst',
                    sweep_inames=sweep_inames,
                    precompute_outer_inames=outer_inames,
                    precompute_inames=(input_prcmpt_iname, 'icell'),
                    temporary_address_space=lp.AddressSpace.LOCAL,
                    default_tag=None,
                    )
            kernel = lp.split_iname(kernel, input_prcmpt_iname,
                    nt, inner_tag="l.0")

    # }}}

    # {{{ Prefetch local operators

    total_shared_vars = []

    if tiling_config.load_mats_to_shared:
        from loopy.transform.data import add_prefetch_for_single_kernel
        #FIXME: Assuming that in all the constants the one with single axis is
        # the one corresponding to quadrature weights. fix it by passing some
        # metadata from TSFC.
        # FIXME: Sweep inames depends on the parallelization strategies for
        # both the matvecs, that needs to be taken care of.

        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()

        for istage, (tr, tc) in enumerate(op_tile_descrs):
            sweep_inames = ('irow{}_inner'.format(istage),
                    'icol{}_inner'.format(istage))
            fetch_outer_inames = 'iblock,icoltile{0},irowtile{0}'.format(
                    istage)
            prefetch_inames = [vng("iprftch") for _ in range(2)]

            for i_op_pos, prftch_from in enumerate(matvec_stage_to_op_matrices[istage]):
                prftch_into = vng('matvec%d_cnst_mtrix_prftch' % istage)
                total_shared_vars.append(prftch_into)

                kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                        var_name=prftch_from,
                        sweep_inames=sweep_inames,
                        temporary_address_space=lp.AddressSpace.LOCAL,
                        dim_arg_names=prefetch_inames,
                        temporary_name=prftch_into,
                        compute_insn_id=ing("prftch_matvec%d" % istage),
                        fetch_outer_inames=fetch_outer_inames,
                        default_tag=None,
                        within='tag:matvec%d' % istage)

                new_temps = kernel.temporary_variables.copy()

                lx, ly = kernel.temporary_variables[prftch_into].shape
                assert lx*ly == tr*tc
                new_temps[prftch_into] = (
                        kernel.temporary_variables[prftch_into].copy(
                            base_storage='prftch_matrix_base',
                            offset=i_op_pos*tr*tc,
                            shape=((i_op_pos+1)*lx, ly)
                            ))

                kernel = kernel.copy(temporary_variables=new_temps)

            kernel = lp.add_dependency(kernel,
                    'tag:matvec%d and (tag:quad_redn or tag:basis_redn)' % istage,
                    'id:prftch_matvec%d*' % istage)
            kernel = lp.add_nosync(kernel, source='id:prftch_matvec%d*' % istage,
                    sink='id:prftch_matvec%d*' % istage,
                    scope='local', empty_ok=True, force=True)

            kernel = lp.join_inames(kernel, prefetch_inames,
                    new_iname='i_matvec%d_prftch' % istage)
            kernel = lp.split_iname(kernel, 'i_matvec%d_prftch' % istage,
                    nc*nt, outer_tag="unr")
            kernel = lp.split_iname(kernel,
                    'i_matvec%d_prftch_inner' % istage,
                    nt, inner_tag='l.0', outer_tag='l.1')

        # {{{ alias shared mem. variables => helps in latency hiding.

        if False:
            # use aliasing instead.
            # TODO: Temporary transformation path until
            # https://gitlab.tiker.net/inducer/loopy/issues/205 is resolved.
            from loopy.transform.data import flatten_variable, absorb_temporary_into
            for prftch_vars in matvec_stage2prftch_vars:
                for var_name in prftch_vars:
                    kernel = flatten_variable(kernel, var_name)

            absorber_temp = []

            for i in range(max(len(k) for k in matvec_stage2prftch_vars)):
                candidate_absorber_names = []
                candidate_absorber_shapes = []
                for prefetch_vars, (tr, tc) in zip(matvec_stage2prftch_vars,
                        op_tile_descrs):
                    if i < len(prefetch_vars):
                        candidate_absorber_names.append(prefetch_vars[i])
                        candidate_absorber_shapes.append(tr*tc)

                absorber_temp.append(candidate_absorber_names[np.argmax(candidate_absorber_shapes)])

            for prftch_vars in matvec_stage2prftch_vars:
                for absorbee, absorber in zip(prftch_vars, absorber_temp):
                    if absorber != absorbee:
                        kernel = absorb_temporary_into(kernel, absorber, absorbee)
        else:
            pass
            # kernel = lp.alias_temporaries(kernel, total_shared_vars)

        # }}}

        # {{{ Adding dependency between the prefetch instructions

        # We have already added the depedencies earlier.
        # So probably this is good to go.

        # FIXME: See if this is necessary...
        # we already enforce dependencies between different matvecs.
        for i in range(len(inputDoFs)):
            kernel = lp.add_dependency(kernel, 'id:prftch_matvec%d*' % (i+1),
                    'tag:quad_wrap_up and tag:matvec%d' % i)
        # kernel = lp.add_dependency(kernel, 'tag:quad_redn', 'id:quad_prftch_insn*')
        # kernel = lp.add_dependency(kernel, 'tag:basis_redn', 'id:basis_prftch_insn*')

        # }}}

        # do not enforce any dependency between the basis reductions and the
        # quadrature reductions.

        # FIXME: These should be already handled in the remove_unnecessary_deps
        # part. Confirm and get rid of this part.
        # kernel = lp.remove_dependency(kernel, 'tag:quad_redn', 'tag:quad_redn')
        # kernel = lp.remove_dependency(kernel, 'tag:basis_redn', 'tag:basis_redn')
        # kernel = lp.add_dependency(kernel, 'tag:quad_wrap_up', 'tag:quad_redn')

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
            # sweep_inames = (quad_iname_in_quad_stage+'_inner')
            # fetch_outer_inames = 'irowtile_matvec1, iblock'
        else:
            sweep_inames = ()
            for i in range(len(inputDoFs)):
                sweep_inames += ('irow%d_inner' % i, 'irowtile%d' % i)
            fetch_outer_inames = 'iblock'

        from loopy.transform.data import add_prefetch_for_single_kernel
        kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                var_name=quad_weights,
                sweep_inames=sweep_inames,
                temporary_address_space=lp.AddressSpace.LOCAL,
                dim_arg_names=(quad_weight_prefetch_iname,),
                temporary_name='cnst_quad_weight_prftch',
                compute_insn_id=quad_weight_prefetch_insn,
                fetch_outer_inames=fetch_outer_inames,
                default_tag=None)

        kernel = lp.add_dependency(kernel, "tag:matvec0 and tag:quad_init", "id:%s" %
                quad_weight_prefetch_insn)

        kernel = lp.split_iname(kernel, quad_weight_prefetch_iname,
                nc * nt, outer_tag="unr")
        kernel = lp.split_iname(kernel, quad_weight_prefetch_iname+'_inner',
                nt,
                outer_tag="l.1", inner_tag="l.0")

    # }}}

    # {{{ divide the matvec of each cell across threads

    for i in range(len(op_tile_descrs)):
        kernel = lp.split_iname(kernel, 'irow%d_inner' % i,
                nt, inner_tag="l.0", outer_tag="unr")

    # }}}

    # post splitting, some variables must be privatized to preserve the logic
    for i, (op_mat_col_length, (tr, tc)) in enumerate(zip(n_inputDoFs+[nquad], op_tile_descrs)):
        if tc < op_mat_col_length:
            from loopy.match import parse_match
            only_var_names = [
                    insn.assignee.name
                    for insn in kernel.instructions
                    if parse_match("(tag:quad_init or tag:basis_init) and"
                        " tag:matvec%d" % i)(kernel, insn)]

            kernel = lp.privatize_temporaries_with_inames(kernel,
                    'irow%d_inner_outer' % i,
                    only_var_names=only_var_names)
            kernel = lp.duplicate_inames(kernel,
                    ['irow%d_inner_outer' % i],
                    within='tag:matvec%d and (tag:quad_wrap_up or tag:basis_wrap_up)' % i)
            kernel = lp.duplicate_inames(kernel,
                    ['irow%d_inner_outer' % i],
                    within='tag:matvec%d and (tag:quad_init or tag:basis_init)' % i)
        else:
            kernel = lp.add_inames_to_insn(kernel, 'icoltile%d' % i,
                'tag:matvec%d' % i)

    # {{{ micro-optimizations

    if nt == 1 and not tiling_config.load_mats_to_shared:
        # FIXME: not general enough!
        raise RuntimeError()
        #@TODO: form_insn_19 and form_insn20 aren't general enough!
        kernel = lp.add_nosync(kernel, "local", "id:form_insn_19 or id:form_insn_20",
                "id:form_insn_21")

    # }}}

    kernel = lp.tag_inames(kernel, "icell:l.1, iblock:g.0")

    kernel = lp.remove_unused_inames(kernel)
    kernel = kernel.copy(loop_priority=frozenset())

    return kernel, args_to_make_global

# }}}


# {{{ auto tile

WARP_SIZE = 32


class AutoTiler:
    """
    Helper class to tune the :class:`pyop2.gpu.tile.TilingConfiguration` for
    :func:`pyop2.gpu.tile.tiled_transform`.

    :attr fem_program: An instance of :class:`loopy.program.Program` which is
        the FEM computational kernel to be tuned.

    See the entrypoint :func:`pyop2.gpu.tile.Autotiler.__call__`
    """
    def __init__(self, fem_program, num_candidate_knls):
        self.fem_program = fem_program
        self.num_candidate_knls = num_candidate_knls

    @cached_property
    def nbasis(self):
        return int(lp.symbolic.pw_aff_to_expr(
            self.fem_program.root_kernel.get_iname_bounds('form_i',
                constants_only=True).size))

    @cached_property
    def nquad(self):
        return int(lp.symbolic.pw_aff_to_expr(
                self.fem_program.root_kernel.get_iname_bounds('form_ip',
                    constants_only=True).size))

    @cached_property
    def num_const_matrices(self):
        """
        Returns the number of constant matrices in the FEM kernel.
        """
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
        """
        Returns the number of variables evaluated at the quadrature nodes.
        """
        evaluation_variables = (set().union(*[insn.write_dependency_names() for
            insn in self.fem_program.root_kernel.instructions if 'quadrature' in insn.tags]) &
            set().union(*[insn.read_dependency_names() for insn in
                self.fem_program.root_kernel.instructions if 'basis' in insn.tags]))

        return len(evaluation_variables)

    def get_local_barriers(self, t1_r, t1_c, t2_r, t2_c):
        """
        Returns the number of block level synchronization instructions in a
        single kernel execution.
        """
        return (
                ceil(self.nquad/t1_r) * ceil(self.nbasis/t1_c)
                + ceil(self.nbasis/t2_r) * ceil(self.nquad/t2_c))

    def theoretical_warps_per_sm(self, tiling_config):
        """
        Returns the number of warps residing on an Streaming Multiprocessor.
        """

        cells_per_block = tiling_config.ncells_per_block
        threads_per_cell = tiling_config.nthreads_per_cell
        t1_r, t1_c = tiling_config.matvec1_row_tile_length, tiling_config.matvec1_col_tile_length
        t2_r, t2_c = tiling_config.matvec2_row_tile_length, tiling_config.matvec2_col_tile_length

        # {{{ computing shared mem usage per block

        shared_usage = (
                self.num_const_matrices*max(t1_r*t1_c, t2_r*t2_c)
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

    def get_work_efficiency(self, tiling_config):
        """
        Returns the efficieny(as a fraction) for a tile defined by t1_r x t1_c,
        t2_r x t2_c.

        One reason for inefficiency is if the number of threads in a CUDA block
        aren't a multiple of the warp size.
        """
        cells_per_block = tiling_config.ncells_per_block
        threads_per_cell = tiling_config.nthreads_per_cell
        t1_r = tiling_config.matvec1_row_tile_length
        t2_r = tiling_config.matvec2_row_tile_length

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

    def actual_warps_per_sm(self, tiling_config):
        """
        Returns "actual warps residing per SM" = Efficiency * "theoretical
        warps reising per SM".
        """
        return (
                self.theoretical_warps_per_sm(tiling_config)
                * self.get_work_efficiency(tiling_config))

    @memoize_method
    def estimated_exec_time(self, tiling_config):
        """
        Returns a metric proportional to the execution time for a
        configuration.
        """

        n_c = tiling_config.ncells_per_block
        n_t = tiling_config.nthreads_per_cell
        t1_r, t1_c = tiling_config.matvec1_row_tile_length, tiling_config.matvec1_col_tile_length
        t2_r, t2_c = tiling_config.matvec2_row_tile_length, tiling_config.matvec2_col_tile_length
        n_w = self.actual_warps_per_sm(tiling_config)

        if n_w == 0:
            return float("inf")
        n_lb = self.get_local_barriers(t1_r, t1_c, t2_r, t2_c)
        n_blocks = (n_w * 32)/(n_t*n_c)

        # nb, nq = self.nbasis, self.nquad
        # return (n_t*nb + nb*nq/(n_t*n_c) + nb*nq*(n_t+n_c)/20.0)/n_w
        return n_lb/n_blocks

    def get_candiate_configs(self):

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
                    if (self.estimated_exec_time(TilingConfiguration(cells, threads,
                        *tile, False, False, True, True, False, False)) < self.estimated_exec_time(
                            TilingConfiguration(best_cells, threads, *tile,
                                False, False, True, True, False, False))):
                        best_cells = cells

                if best_cells != 10000:
                    params.append(TilingConfiguration(best_cells, threads, *tile,
                                False, False, True, True, False, False))

        # sort the parameters with highest occupancy.
        params.sort(key=lambda P:  self.estimated_exec_time(P))

        return params[:self.num_candidate_knls]

    @memoize_method
    def convert_numpy_arrays_to_cuda_mems(self, ary):
        ary = np.array(ary)
        ary_gpu = cuda.mem_alloc(ary.nbytes)
        cuda.memcpy_htod(src=ary, dest=ary_gpu)
        return ary_gpu

    def __call__(self, args, argshapes):

        best_performing_time = float("inf")
        best_performing_config = None
        nrounds = 15
        nwarmup = 5

        copied_args = args[:2]
        for i, arg in enumerate(self.fem_program.args[2:]):
            if arg.name in self.fem_program.root_kernel.get_written_variables():
                # arg is written during kernel execution => make a copy
                arg_gpu = cuda.mem_alloc(
                        int(np.prod(argshapes[i])*arg.dtype.itemsize))
                cuda.memcpy_dtod(src=args[i+2], dest=arg_gpu,
                        size=int(np.prod(argshapes[i])*arg.dtype.itemsize))
                copied_args += (arg_gpu,)
            else:
                # arg is read only => pass the same arg to the knl
                copied_args += (args[i+2],)

        from pyop2.gpu.tile import tiled_transform

        for tiling_config in self.get_candiate_configs():
            kernel, extra_args = tiled_transform(
                    self.fem_program.root_kernel, self.fem_program.callables_table,
                    tiling_config)
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
                tiling_config, exec_time))

            if exec_time < best_performing_time:
                best_performing_time = exec_time
                best_performing_config = tiling_config

        return tiled_transform(
                self.fem_program.root_kernel, self.fem_program.callables_table,
                best_performing_config)

# }}}

# vim: fdm=marker
