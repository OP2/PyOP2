# This file contains classes that manipulate a kernel's Abstract Syntax Tree
# Typical functionalities include parsing from C code, code generation, etc.


def generate_code(node):
    """Generate code for the tree rooted in node"""

    return node.gencode()
