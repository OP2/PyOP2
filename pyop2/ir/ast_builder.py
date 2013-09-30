# This file contains methods that help building a kernel's Abstract Syntax Tree

from ast_base import *


def add_node(parent, children):
    """Add the given list of children a node"""
    parent.children = children


def create_sum(operands, is_leaf):
    """If is_lead is false then wrap the sum with parentheses"""
    new_sym = Sum()
    new_sym.children = [Symbol(sym) for sym in operands]
    if not is_leaf:
        par = Parentheses()
        par.children.append(new_sym)
        new_sym = par
    return new_sym
