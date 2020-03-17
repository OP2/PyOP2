import loopy as lp


def snpt_transform(kernel, block_size):
    """
    SNPT := Single 'n' Per Thread.

    Implements outer-loop parallelization strategy.

    PyOP2 uses 'n' as the outer loop iname. In Firedrake 'n' might denote
    either a cell or a DOF.
    """

    kernel = lp.assume(kernel, "start < end")
    kernel = lp.split_iname(kernel, "n", block_size,
            outer_tag="g.0", inner_tag="l.0")

    # {{{ making consts as globals

    args_to_make_global = []
    pack_consts_to_globals = False

    if pack_consts_to_globals:
        args_to_make_global = [tv.initializer.flatten()
                for tv in kernel.temporary_variables.values()
                if (tv.initializer is not None
                    and tv.address_space == lp.AddressSpace.GLOBAL)]

        new_temps = dict((tv.name, tv.copy(initializer=None))
                if (tv.initializer is not None
                    and tv.address_space == lp.AddressSpace.GLOBAL)
                else (tv.name, tv) for tv in
                kernel.temporary_variables.values())

        kernel = kernel.copy(temporary_variables=new_temps)

    # }}}

    return kernel, args_to_make_global
