# Test the AST data structures and routines

from ast_base import *
from ast_builder import create_sum, c_sym


def main():
    print "Creating a fake AST..."
    exp_1 = create_simple_exp()
    exp_2 = create_array_exp()

    ass_1 = Assign(Symbol("A", "j"), exp_1)
    decl_1 = Decl("double", Symbol("B", (3,)), qualifiers=["const"])

    i_sym = Symbol("i", ())
    for_1 = For(Decl("int", i_sym, c_sym(0)), Less(i_sym, c_sym(3)),
                Incr(i_sym, c_sym(1)), Block([decl_1], open_scope=True))

    decl_2 = Decl("double", Symbol("B", (3,)),
                  ArrayInit("{3, 3, 3}"), ["const"])

    for_2 = For(Decl("int", i_sym, c_sym(0)), Less(i_sym, c_sym(3)),
                Incr(i_sym, c_sym(1)), Block([decl_1, for_1], open_scope=True))

    root = Root([ass_1, exp_2, for_1, decl_2, for_2])

    print "Generate the code for this AST"
    print root.gencode()


def create_simple_exp():
    sum1 = create_sum(((2, ()), (3, ())), False)
    sum2 = create_sum(((4, ()), (5, ())), False)
    return Sum(sum1, sum2)


def create_array_exp():
    return create_sum((("A", ("j", "k")), ("B", ("j", "k"))), True)


if __name__ == "__main__":
    main()
