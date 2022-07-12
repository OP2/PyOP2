import loopy as lp


def _make_tv_array_arg(tv):
    assert tv.address_space != lp.AddressSpace.PRIVATE
    arg = lp.ArrayArg(name=tv.name,
                      dtype=tv.dtype,
                      shape=tv.shape,
                      dim_tags=tv.dim_tags,
                      offset=tv.offset,
                      dim_names=tv.dim_names,
                      order=tv.order,
                      alignment=tv.alignment,
                      address_space=tv.address_space,
                      is_output=not tv.read_only,
                      is_input=tv.read_only)
    return arg


def split_n_across_workgroups(kernel, workgroup_size):
    """
    Returns a transformed version of *kernel* with the workload in the loop
    with induction variable 'n' distributed across work-groups of size
    *workgroup_size* and each work-item in the work-group performing the work
    of a single iteration of 'n'.
    """

    kernel = lp.assume(kernel, "start < end")
    kernel = lp.split_iname(kernel, "n", workgroup_size,
                            outer_tag="g.0", inner_tag="l.0")

    # {{{ making consts as globals: necessary to make the strategy emit valid
    # kernels for all forms

    old_temps = kernel.temporary_variables.copy()
    args_to_make_global = [tv.initializer.flatten()
                           for tv in old_temps.values()
                           if tv.initializer is not None]

    new_temps = {tv.name: tv
                 for tv in old_temps.values()
                 if tv.initializer is None}
    kernel = kernel.copy(args=kernel.args+[_make_tv_array_arg(tv)
                                           for tv in old_temps.values()
                                           if tv.initializer is not None],
                         temporary_variables=new_temps)

    # }}}

    return kernel, args_to_make_global
