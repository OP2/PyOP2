# This file contains methods that help building a kernel's Abstract Syntax Tree

from ast_base import *


def add_node(parent, children):
    """Add the given list of children a node"""
    for child in children:
        parent.children.append(child)


def create_sum(operands, is_leaf):
    """If is_lead is false then wrap the sum with parentheses"""
    op1 = operands[0]
    op2 = operands[1]
    new_sym = Sum(Symbol(op1[0], op1[1]), Symbol(op2[0], op2[1]))
    if not is_leaf:
        par = Par(new_sym)
        new_sym = par
    return new_sym
