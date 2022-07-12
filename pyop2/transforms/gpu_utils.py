import loopy as lp
from pyop2.configuration import configuration


def get_loopy_target(target):
    if target == "opencl":
        return lp.PyOpenCLTarget()
    elif target == "cuda":
        return lp.CudaTarget()
    else:
        raise NotImplementedError()


def preprocess_t_unit_for_gpu(t_unit):

    # {{{ inline all kernels in t_unit

    kernels_to_inline = {
        name for name, clbl in t_unit.callables_table.items()
        if isinstance(clbl, lp.CallableKernel)}

    for knl_name in kernels_to_inline:
        t_unit = lp.inline_callable_kernel(t_unit, knl_name)

    # }}}

    kernel = t_unit.default_entrypoint

    # changing the address space of temps
    def _change_aspace_tvs(tv):
        if tv.read_only:
            assert tv.initializer is not None
            return tv.copy(address_space=lp.AddressSpace.GLOBAL)
        else:
            return tv.copy(address_space=lp.AddressSpace.PRIVATE)

    new_tvs = {tv_name: _change_aspace_tvs(tv) for tv_name, tv in
               kernel.temporary_variables.items()}
    kernel = kernel.copy(temporary_variables=new_tvs)

    def insn_needs_atomic(insn):
        # updates to global variables are atomic
        import pymbolic
        if isinstance(insn, lp.Assignment):
            if isinstance(insn.assignee, pymbolic.primitives.Subscript):
                assignee_name = insn.assignee.aggregate.name
            else:
                assert isinstance(insn.assignee, pymbolic.primitives.Variable)
                assignee_name = insn.assignee.name

            if assignee_name in kernel.arg_dict:
                return assignee_name in insn.read_dependency_names()
        return False

    new_insns = []
    args_marked_for_atomic = set()
    for insn in kernel.instructions:
        if insn_needs_atomic(insn):
            atomicity = (lp.AtomicUpdate(insn.assignee.aggregate.name), )
            insn = insn.copy(atomicity=atomicity)
            args_marked_for_atomic |= set([insn.assignee.aggregate.name])

        new_insns.append(insn)

    # label args as atomic
    new_args = []
    for arg in kernel.args:
        if arg.name in args_marked_for_atomic:
            new_args.append(arg.copy(for_atomic=True))
        else:
            new_args.append(arg)

    kernel = kernel.copy(instructions=new_insns, args=new_args)

    return t_unit.with_kernel(kernel)


def apply_gpu_transforms(t_unit, target):
    t_unit = t_unit.copy(target=get_loopy_target(target))
    t_unit = preprocess_t_unit_for_gpu(t_unit)
    kernel = t_unit.default_entrypoint
    transform_strategy = configuration["gpu_strategy"]

    kernel = lp.assume(kernel, "end > start")

    if transform_strategy == "snpt":
        from pyop2.transforms.snpt import split_n_across_workgroups
        kernel, args_to_make_global = split_n_across_workgroups(kernel, 32)
    else:
        raise NotImplementedError(f"'{transform_strategy}' transform strategy.")

    t_unit = t_unit.with_kernel(kernel)

    return t_unit, args_to_make_global
