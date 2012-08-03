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

"""OP2 exception types"""

class DataTypeError(TypeError):
    """Invalid type for data."""

class DimTypeError(TypeError):
    """Invalid type for dimension."""

class IndexTypeError(TypeError):
    """Invalid type for index."""

class NameTypeError(TypeError):
    """Invalid type for name."""

class SetTypeError(TypeError):
    """Invalid type for Set."""

class SizeTypeError(TypeError):
    """Invalid type for size."""

class SparsityTypeError(TypeError):
    """Invalid type for sparsity."""

class MapTypeError(TypeError):
    """Invalid type for map."""

class MatTypeError(TypeError):
    """Invalid type for mat."""

class DatTypeError(TypeError):
    """Invalid type for dat."""

class DataValueError(ValueError):
    """Illegal value for data."""

class IndexValueError(ValueError):
    """Illegal value for index."""

class ModeValueError(ValueError):
    """Illegal value for mode."""

class SetValueError(ValueError):
    """Illegal value for Set."""
