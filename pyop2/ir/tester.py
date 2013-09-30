# Test the AST data structures and routines

from ast_base import *
from ast_builder import add_node, create_sum
from ast_handler import generate_code


def main():
    print "Creating a fake AST..."
    root = Root()
    top_sum = Sum()
    add_node(root, [top_sum])
    sum1 = create_sum([2, 3], False)
    sum2 = create_sum([4, 5], False)
    add_node(top_sum, [sum1, sum2])

    print "Generate the code for this AST"
    print generate_code(root)

if __name__ == "__main__":
    main()
