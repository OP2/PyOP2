import loopy as lp


def extrude_transform(kernel, cells_per_block, layers_per_block):
    from loopy.loop import fuse_loop_domains
    kernel = lp.split_iname(kernel, "n", cells_per_block, inner_tag="l.0",
            outer_tag="g.0")
    kernel = fuse_loop_domains(kernel)
    kernel = lp.assignment_to_subst(kernel, 't1')
    kernel = lp.assignment_to_subst(kernel, 't2')
    kernel = lp.expand_subst(kernel)
    print(kernel)
    1/0
    if False:
        # FIXME: splitting "layer" gives a loopy error
        kernel = lp.split_iname(kernel, "layer", layers_per_block, inner_tag="l.1",
                outer_tag="g.1")
    return kernel, ()
