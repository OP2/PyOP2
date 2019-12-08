import loopy as lp


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


def work_which_should_be_done_by_passing_metadata(kernel,
        output_basis_coeff_temp, quad_iname):
    from pymbolic.primitives import Variable

    # {{{ scatter iname

    scatter_insn, = [insn for insn in kernel.instructions if 'scatter' in
            insn.tags]
    scatter_map = scatter_insn.assignee.index_tuple[0]
    scatter_iname, = set(scatter_map.index_tuple) - set([Variable('n')])
    scatter_iname = scatter_iname.name

    # }}}

    # {{{ basis init iname

    basis_gather_insn, = [insn for insn in kernel.instructions if 'gather' in
            insn.tags and output_basis_coeff_temp in
            insn.write_dependency_names()]
    basis_gather_iname = basis_gather_insn.assignee.index_tuple[1].name

    # }}}

    basis_redn_insn = [insn for insn in kernel.instructions if 'basis' in
            insn.tags][0]
    basis_iname_in_basis_redn, = basis_redn_insn.within_inames - frozenset(['n', quad_iname])

    return basis_gather_iname, scatter_iname, basis_iname_in_basis_redn


def tiled_transform(kernel, callables_table, ncells_per_block,
        nthreads_per_cell,
        matvec1_row_tile_length, matvec1_col_tile_length,
        matvec2_row_tile_length, matvec2_col_tile_length,
        load_coordinates_to_shared,
        load_input_to_shared,
        load_mats_to_shared,
        load_quad_weights_to_shared,
        tiled_prefetch_of_inputs,
        tiled_prefetch_of_quad_weights):
    """
    Matvec1 is the function evaluation part at the quad points.
    Matvec2 is the basis coefficients computation part.
    """

    # {{{ FIXME: Setting names which should be set by TSFC

    quad_iname = 'form_ip'
    input_basis_coeff_temp = 't0'
    coords_temp = 't1'
    output_basis_coeff_temp = 't2'
    basis_init_iname, scatter_iname, basis_iname_in_basis_redn = (
            work_which_should_be_done_by_passing_metadata(kernel,
                output_basis_coeff_temp, quad_iname))
    quad_iname_in_basis_redn = 'form_ip_basis'
    quad_iname_in_quad_redn = 'form_ip_quad'
    basis_iname_in_quad_redn = 'form_i'
    input_basis_coeff_subst = input_basis_coeff_temp+'_subst'

    # }}}

    # {{{ reading info about the finite element

    nquad = int(lp.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds('form_ip', constants_only=True).size))
    nbasis = int(lp.symbolic.pw_aff_to_expr(
            kernel.get_iname_bounds(basis_iname_in_basis_redn, constants_only=True).size))

    # }}}

    # {{{ tagging the stages of the kernel

    #TODO: Should be interpreted in TSFC

    new_insns = []

    done_with_jacobi_eval = False
    done_with_quad_init = False
    done_with_quad_reduction = False
    done_with_quad_wrap_up = False
    done_with_basis_reduction = False

    for insn in kernel.instructions:
        if not done_with_jacobi_eval:
            if 'form_ip' in insn.within_inames:
                done_with_jacobi_eval = True

            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["jacobi_eval"])))
                continue
        if not done_with_quad_init:
            if 'form_i' in insn.within_inames:
                done_with_quad_init = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_init"])))
                continue
        if not done_with_quad_reduction:
            if 'form_i' not in insn.within_inames:
                done_with_quad_reduction = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_redn"])))
                continue
        if not done_with_quad_wrap_up:
            if 'basis' in insn.tags:
                done_with_quad_wrap_up = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["quad_wrap_up"])))
                continue
        if not done_with_basis_reduction:
            if 'form_ip' not in insn.within_inames:
                done_with_basis_reduction = True
            else:
                new_insns.append(insn.copy(tags=insn.tags
                    | frozenset(["basis_redn"])))
                continue
        new_insns.append(insn)

    assert done_with_basis_reduction

    kernel = kernel.copy(instructions=new_insns)

    # }}}

    # {{{ privatize temps for function evals and make them LOCAL

    #FIXME: Need these variables from TSFC's metadata
    evaluation_variables = (set().union(*[insn.write_dependency_names() for insn in kernel.instructions if 'quad_wrap_up' in insn.tags])
            & set().union(*[insn.read_dependency_names() for insn in kernel.instructions if 'basis' in insn.tags]))

    kernel = lp.privatize_temporaries_with_inames(kernel, 'form_ip',
            evaluation_variables)
    new_temps = kernel.temporary_variables.copy()
    for eval_var in evaluation_variables:
        new_temps[eval_var] = new_temps[eval_var].copy(
                address_space=lp.AddressSpace.LOCAL)
    kernel = kernel.copy(temporary_variables=new_temps)

    # Duplicate inames to separate transformation logic for quadrature and basis part
    kernel = lp.duplicate_inames(kernel, quad_iname, "tag:quadrature",
            quad_iname_in_quad_redn)
    kernel = lp.duplicate_inames(kernel, quad_iname, "tag:basis",
            quad_iname_in_basis_redn)

    # }}}

    # {{{ change address space of constants to '__global'

    old_temps = kernel.temporary_variables.copy()
    args_to_make_global = [tv.initializer.flatten() for tv in old_temps.values() if tv.initializer is not None]

    new_temps = dict((tv.name, tv) for tv in old_temps.values() if tv.initializer is None)
    kernel = kernel.copy(
            args=kernel.args+[_make_tv_array_arg(tv) for tv in old_temps.values() if tv.initializer is not None],
            temporary_variables=new_temps)

    # }}}

    from lp.loop import fuse_loop_domains
    kernel = fuse_loop_domains(kernel)

    from lp.transform.data import remove_unused_axes_in_temporaries
    kernel = remove_unused_axes_in_temporaries(kernel)

    # {{{ remove noops

    noop_insns = set([insn.id for insn in kernel.instructions if
            isinstance(insn, lp.NoOpInstruction)])
    kernel = lp.remove_instructions(kernel, noop_insns)

    # }}}

    # Realize CUDA blocks
    kernel = lp.split_iname(kernel, "n", ncells_per_block,
            outer_iname="iblock", inner_iname="icell")

    from lp.transform.batch import save_temporaries_in_loop
    kernel = save_temporaries_in_loop(kernel, 'icell',
            evaluation_variables)

    #FIXME: Do not use hard-coded inames, this change should also be in TSFC.
    # We need this statement because we have to cut down the size of the number
    # of basis coeffs controlled by each thread(if there are multiple threads)
    kernel = lp.rename_iname(kernel, scatter_iname,
            basis_iname_in_basis_redn, True)
    kernel = lp.rename_iname(kernel, basis_init_iname,
            basis_iname_in_basis_redn, True)

    from lp.transform.instruction import remove_unnecessary_deps
    kernel = remove_unnecessary_deps(kernel)

    from lp.transform.make_scalar import remove_axis
    kernel = remove_axis(kernel, output_basis_coeff_temp, 0)

    kernel = lp.add_dependency(kernel,
            'writes:{}'.format(output_basis_coeff_temp),
            'tag:quad_wrap_up')

    if load_coordinates_to_shared:
        # FIXME: This seems unnecessary as of now. I might choose to not
        # support it.
        kernel = lp.privatize_temporaries_with_inames(kernel, 'icell',
                [coords_temp])
        kernel = lp.assignment_to_subst(kernel, coords_temp)
        raise NotImplementedError("This might be only useful for high order"
                " meshes.")

    # Splitting for tiles in matvec1
    kernel = lp.split_iname(kernel, quad_iname_in_quad_redn, matvec1_row_tile_length, outer_iname='irowtile_matvec1')
    kernel = lp.split_iname(kernel, basis_iname_in_quad_redn, matvec1_col_tile_length, outer_iname='icoltile_matvec1')

    # Splitting for tiles in matvec2
    kernel = lp.split_iname(kernel, basis_iname_in_basis_redn, matvec2_row_tile_length, outer_iname='irowtile_matvec2')
    kernel = lp.split_iname(kernel, quad_iname_in_basis_redn, matvec2_col_tile_length, outer_iname='icoltile_matvec2')

    # {{{ Prefetch wizardry

    if load_input_to_shared:
        from lp.transform.precompute import precompute_for_single_kernel
        kernel = save_temporaries_in_loop(kernel, 'icell',
                [input_basis_coeff_temp])
        kernel = lp.assignment_to_subst(kernel, input_basis_coeff_temp)
        input_prcmpt_iname = 'input_basis_prcmpt'
        if tiled_prefetch_of_inputs:
            sweep_inames = ('icell', basis_iname_in_quad_redn+'_inner')
            outer_inames = 'iblock,icoltile_matvec1,irowtile_matvec1'
        else:
            sweep_inames = ('icell', 'icoltile_matvec1', basis_iname_in_quad_redn+'_inner')
            outer_inames = 'iblock'
        kernel = precompute_for_single_kernel(kernel, callables_table,
                subst_use=input_basis_coeff_subst,
                sweep_inames=sweep_inames,
                precompute_outer_inames=outer_inames,
                precompute_inames=('icell', input_prcmpt_iname),
                temporary_address_space=lp.AddressSpace.LOCAL,
                default_tag=None,
                )
        kernel = lp.split_iname(kernel, input_prcmpt_iname,
                nthreads_per_cell, inner_tag="l.0")

    if load_mats_to_shared:
        from lp.transform.data import add_prefetch_for_single_kernel
        #FIXME: Assuming that in all the constants the one with single axis is
        # the one corresponding to quadrature weights. fix it by passing some
        # metadata from TSFC.
        # FIXME: Sweep inames depends on the parallelization strategies for
        # both the matvecs, that needs to be taken care of.
        const_matrices_names = set([tv.name for tv in old_temps.values() if tv.initializer is not None and len(tv.shape)>1])

        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()

        # {{{ Prefetching: QUAD PART

        quad_const_matrices = const_matrices_names & frozenset().union(*[insn.read_dependency_names() for insn in
            kernel.instructions if 'quad_redn' in insn.tags])
        sweep_inames = (quad_iname_in_quad_redn+'_inner',
                basis_iname_in_quad_redn+'_inner')
        fetch_outer_inames = 'iblock,icoltile_matvec1,irowtile_matvec1'

        quad_prefetch_insns = []

        quad_temp_names = [vng('quad_cnst_mtrix_prftch') for _ in quad_const_matrices]
        prefetch_inames = [vng("iprftch") for _ in range(2)]
        for temp_name, var_name in zip(quad_temp_names, quad_const_matrices):
            quad_prefetch_insns.append(ing("quad_prftch_insn"))

            kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                    var_name=var_name,
                    sweep_inames=sweep_inames,
                    temporary_address_space=lp.AddressSpace.LOCAL,
                    dim_arg_names=prefetch_inames,
                    temporary_name=temp_name,
                    compute_insn_id=quad_prefetch_insns[-1],
                    fetch_outer_inames=fetch_outer_inames,
                    default_tag=None,
                    within="tag:quad_redn")

        #FIXME: In order to save on compilation time we are not sticking to
        # coalesced accesses Otherwise we should join the following inames and
        # then split into nthreads_per_cell
        kernel = lp.join_inames(kernel, prefetch_inames,
                new_iname='quad_prftch_iname')
        kernel = lp.split_iname(kernel, 'quad_prftch_iname',
                ncells_per_block*nthreads_per_cell, outer_tag="ilp")
        kernel = lp.split_iname(kernel, 'quad_prftch_iname_inner',
                nthreads_per_cell, inner_tag='l.0', outer_tag='l.1')

        # kernel = lp.split_iname(kernel, prefetch_inames[1],
        #         nthreads_per_cell, inner_tag="l.0", outer_tag="ilp")
        # kernel = lp.split_iname(kernel, prefetch_inames[0],
        #         ncells_per_block, inner_tag="l.1", outer_tag="ilp")

        # }}}

        # {{{ Prefetching: BASIS PART

        basis_const_matrices = const_matrices_names & frozenset().union(*[insn.read_dependency_names() for insn in
            kernel.instructions if 'basis_redn' in insn.tags])
        basis_temp_names = [vng('basis_cnst_mtrix_prftch') for _ in basis_const_matrices]

        sweep_inames = (basis_iname_in_basis_redn+'_inner',
                quad_iname_in_basis_redn+'_inner')
        fetch_outer_inames = 'iblock,icoltile_matvec2,irowtile_matvec2'

        basis_prefetch_insns = []
        prefetch_inames = [vng("iprftch") for _ in range(2)]
        for temp_name, var_name in zip(basis_temp_names, basis_const_matrices):
            basis_prefetch_insns.append(ing("basis_prftch_insn"))

            kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                    var_name=var_name,
                    sweep_inames=sweep_inames,
                    temporary_address_space=lp.AddressSpace.LOCAL,
                    dim_arg_names=prefetch_inames,
                    temporary_name=temp_name,
                    compute_insn_id=basis_prefetch_insns[-1],
                    fetch_outer_inames=fetch_outer_inames,
                    default_tag=None,
                    within="tag:basis_redn")

        # See FIXME for the quad part at this point
        kernel = lp.join_inames(kernel, prefetch_inames,
                new_iname='basis_prftch_iname')
        kernel = lp.split_iname(kernel, 'basis_prftch_iname',
                ncells_per_block*nthreads_per_cell, outer_tag="ilp")
        kernel = lp.split_iname(kernel, 'basis_prftch_iname_inner',
                nthreads_per_cell, inner_tag='l.0', outer_tag='l.1')

        # kernel = lp.split_iname(kernel, prefetch_inames[1],
        #         nthreads_per_cell, inner_tag="l.0", outer_tag="ilp")
        # kernel = lp.split_iname(kernel, prefetch_inames[0],
        #         ncells_per_block, inner_tag="l.1", outer_tag="ilp")

        # }}}

        # {{{ using the same variable for both the prefetch shared mems

        from lp.transform.data import flatten_variable, absorb_temporary_into
        for var_name in quad_temp_names+basis_temp_names:
            kernel = flatten_variable(kernel, var_name)
        for quad_temp_name, basis_temp_name in zip(quad_temp_names,
                basis_temp_names):
            if (matvec2_row_tile_length*matvec2_col_tile_length >= matvec1_row_tile_length*matvec1_col_tile_length):
                kernel = absorb_temporary_into(kernel, basis_temp_name, quad_temp_name)
            else:
                kernel = absorb_temporary_into(kernel, quad_temp_name, basis_temp_name)

        # }}}

        # {{{ Adding dependency between the prefetch instructions

        kernel = lp.add_dependency(kernel,
                " or ".join("id:{}".format(insn_id) for insn_id in
                    basis_prefetch_insns), "tag:quadrature")

        kernel = lp.add_dependency(kernel, 'tag:quad_redn', 'id:quad_prftch_insn*')
        kernel = lp.add_dependency(kernel, 'tag:basis_redn', 'id:basis_prftch_insn*')

        # }}}

        # do not enforce any dependency between the basis reductions and the
        # quadrature reductions.

        kernel = lp.remove_dependency(kernel, 'tag:quad_redn', 'tag:quad_redn')
        kernel = lp.remove_dependency(kernel, 'tag:basis_redn', 'tag:basis_redn')
        kernel = lp.add_dependency(kernel, 'tag:quad_wrap_up', 'tag:quad_redn')

    # }}}

    # {{{ Prefetch: Quad Weights

    if load_quad_weights_to_shared:
        from lp.transform.data import add_prefetch_for_single_kernel
        quad_weights, = [tv.name for tv in old_temps.values() if tv.initializer is not None and len(tv.shape) == 1]
        vng = kernel.get_var_name_generator()
        ing = kernel.get_instruction_id_generator()
        quad_weight_prefetch_insn = ing("quad_wt_prftch_insn")
        quad_weight_prefetch_iname = vng("iprtftch")

        if tiled_prefetch_of_quad_weights:
            sweep_inames = (quad_iname_in_quad_redn+'_inner')
            fetch_outer_inames = 'irowtile_matvec1, iblock'
        else:
            sweep_inames = ('irowtile_matvec1', quad_iname_in_quad_redn+'_inner',)
            fetch_outer_inames = 'iblock'

        kernel = add_prefetch_for_single_kernel(kernel, callables_table,
                var_name=quad_weights,
                sweep_inames=sweep_inames,
                temporary_address_space=lp.AddressSpace.LOCAL,
                dim_arg_names=(quad_weight_prefetch_iname,),
                temporary_name='cnst_quad_weight_prftch',
                compute_insn_id=quad_weight_prefetch_insn,
                fetch_outer_inames=fetch_outer_inames,
                default_tag=None,
                within="tag:quad_wrap_up")

        kernel = lp.split_iname(kernel, quad_weight_prefetch_iname,
                ncells_per_block*nthreads_per_cell, outer_tag="ilp")
        kernel = lp.split_iname(kernel, quad_weight_prefetch_iname+'_inner',
                nthreads_per_cell,
                outer_tag="l.1", inner_tag="l.0")

    # }}}

    # {{{ divide matvec1-tile's work across threads

    kernel = lp.split_iname(kernel, quad_iname_in_quad_redn+'_inner',
            nthreads_per_cell, inner_tag="l.0", outer_tag="ilp")

    # }}}

    # {{{ diving matvec2-tile's work across threads

    kernel = lp.split_iname(kernel, basis_iname_in_basis_redn+'_inner',
            nthreads_per_cell, inner_tag="l.0", outer_tag="ilp")

    # }}}

    if matvec1_col_tile_length < nbasis:
        only_var_names = [insn.assignee.name for insn in kernel.instructions if
                'quad_init' in insn.tags]
        kernel = lp.privatize_temporaries_with_inames(kernel,
                quad_iname_in_quad_redn+'_inner_outer',
                only_var_names=only_var_names)
        kernel = lp.duplicate_inames(kernel,
                [quad_iname_in_quad_redn+'_inner_outer', ],
                within='tag:quad_wrap_up')
        kernel = lp.duplicate_inames(kernel,
                [quad_iname_in_quad_redn+'_inner_outer'],
                'tag:quad_init')
    else:
        kernel = lp.add_inames_to_insn(kernel, 'icoltile_matvec1', 'tag:quad_wrap_up or tag:quad_init')

    # before this point 't2' should be made a scalar.

    if matvec2_col_tile_length < nquad:
        #@TODO; 't2' is not generalized enough.
        kernel = lp.privatize_temporaries_with_inames(kernel,
                basis_iname_in_basis_redn+'_inner_outer',
                only_var_names=[output_basis_coeff_temp])
        kernel = lp.duplicate_inames(kernel, [basis_iname_in_basis_redn+'_inner_outer'], within='tag:scatter')
        kernel = lp.duplicate_inames(kernel,
                [basis_iname_in_basis_redn+'_inner_outer'],
                within='tag:gather and writes:{}'.format(output_basis_coeff_temp))
    else:
        kernel = lp.add_inames_to_insn(kernel, 'icoltile_matvec2',
                'tag:scatter or (tag:gather and writes:{})'.format(output_basis_coeff_temp))

    # {{{ micro-optimizations

    if nthreads_per_cell == 1 and not load_mats_to_shared:
        #@TODO: form_insn_19 and form_insn20 aren't general enough!
        kernel = lp.add_nosync(kernel, "local", "id:form_insn_19 or id:form_insn_20",
                "id:form_insn_21")

    # }}}

    kernel = lp.tag_inames(kernel, "icell:l.1, iblock:g.0")

    kernel = lp.remove_unused_inames(kernel)
    kernel = kernel.copy(loop_priority=frozenset())

    return kernel, args_to_make_global

