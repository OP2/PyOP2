# This file is part of PyOP2
#
# PyOP2 is Copyright (c) 2012, Imperial College London and
# others. Please see the AUTHORS file in the main source directory for
# a full list of copyright holders.  All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
#     * Redistributions of source code must retain the above copyright
#       notice, this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright
#       notice, this list of conditions and the following disclaimer in the
#       documentation and/or other materials provided with the distribution.
#     * The name of Imperial College London or that of other
#       contributors may not be used to endorse or promote products
#       derived from this software without specific prior written
#       permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTERS
# ''AS IS'' AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDERS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
# INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
# HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT,
# STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED
# OF THE POSSIBILITY OF SUCH DAMAGE.

"""
This is a standard library of commonly-performed operations on
OP2 data structures.
"""

import op2

# Tests: evaluating 1D and 2D expressions

def eval_expr(expr, v, x)
    """Evaluate a function of x on v. x and v may not be the same dat."""
    code = "void expr(%s *x, %s *v) { *v = %s; }" %
               (x.ctype, v.ctype, expr)
    kernel = op2.Kernel(code, "expr")
    op2.par_loop(kernel, dat.dataset,
                 coords(op2.IdentityMap, op2.READ),
                 dat(op2.IdentityMap, op2.WRITE))

# Tests - zeroing 1D and 2D dats

def zero(dat):
    """Zero all the elements of a dat"""
    lines = [ ("d[%d] = 0; " % d) for d in xrange(dat.dim) ]
    code = "void zero(%s *d) { %s }" % (dat.ctype, lines)
    kernel = op2.Kernel(code, "zero")
    op2.par_loop(kernel, dat.dataset,
                 dat(op2.IdentityMap, op2.WRITE))

def binop(dat1, dat2, op, dat3=None)
    """Compute dat1 op dat2. If dat3 is None, the result is stored in dat1."""
    if dat3 is None:
        code = "void binop(%s d1, %s d2){ "
    # Needs completion

def add(dat, addand):
    pass

def sub(dat, suband):
    pass

def mul(dat, multiplicand):
    pass

def div(dat, divisor):
    pass
