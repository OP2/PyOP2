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

"""OP2 wraper composer for the sequential and OpenMP backends."""


from configuration import configuration

store_array = """
void storeArray_%(type)s(%(type)s *a, int len, char* filename){
  char path[80];
  path[0] = \'\\0\';
  strcat(path, \"/tmp/\");
  strcat(path, filename);
  strcat(path, \".bin\");
  FILE* fp = fopen(path, \"wb\");
  fwrite(a, sizeof(%(type)s), len, fp);
  fclose(fp);
}
"""


def compose_wrapper(backend="sequential"):
    wrapper = ""

    if configuration["hpc_save_result"]:
        wrapper += store_array % {"type": "int"}
        wrapper += store_array % {"type": "double"}

    if backend == "sequential":
        wrapper += """
    double %(wrapper_name)s(int start, int end,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(const_args)s
                      %(layer_arg)s"""

    elif backend == "openmp":
    	wrapper += """
    double %(wrapper_name)s(int boffset,
                      int nblocks,
                      int *blkmap,
                      int *offset,
                      int *nelems,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(const_args)s
                      %(layer_arg)s
    """
    if configuration["hpc_profiling"]:
        wrapper += """
        %(other_args)s"""

    wrapper += ") {"

    if configuration["papi_flops"]:
        wrapper += """
        %(papi_decl)s;
        %(papi_init)s;
        """

    wrapper += """
        %(user_code)s
        %(wrapper_decs)s;
        %(const_inits)s;
        """

    if backend == "sequential":
        wrapper += """
        %(map_decl)s
        %(vec_decs)s;
        """

    if configuration["hpc_profiling"]:
        wrapper += """
        %(timer_declare)s
        %(timer_start)s
        """

    if configuration["times"] > 1:
        wrapper += """
        %(times_loop_start)s
        """

    if backend == "openmp":
        wrapper += """
        #pragma omp parallel shared(boffset, nblocks, nelems, blkmap)
        {
          %(map_decl)s
          int tid = omp_get_thread_num();
          %(interm_globals_decl)s;
          %(interm_globals_init)s;
          %(vec_decs)s;

          #pragma omp for schedule(static)
          for ( int __b = boffset; __b < boffset + nblocks; __b++ )
          {
            int bid = blkmap[__b];
            int nelem = nelems[bid];
            int efirst = offset[bid];
            for (int n = efirst; n < efirst+ nelem; n++ )
            {
        """

    if backend == "sequential":
        wrapper += """
        for ( int n = start; n < end; n++ ) {
        """

    wrapper += """
        int i = %(index_expr)s;
        %(vec_inits)s;
        %(map_init)s;
        %(extr_loop)s
        %(map_bcs_m)s;
        """

    if configuration["iaca"]:
        wrapper += """
        %(iaca_start)s
        """
    wrapper += """
        %(buffer_decl)s;
        %(buffer_gather)s

        %(kernel_name)s(%(kernel_args)s);
    """

    if configuration["hpc_debug"]:
        wrapper += """
        %(print_contrib)s
        """

    wrapper += """
        %(itset_loop_body)s
        %(map_bcs_p)s;
        %(apply_offset)s;
        %(extr_loop_close)s
    """

    if configuration["iaca"]:
        wrapper += """
        %(iaca_end)s
        """

    wrapper += "}"

    if backend == "openmp":
        wrapper += """
            }
        %(interm_globals_writeback)s;
      }
        """

    if configuration["times"] > 1:
        wrapper += """
        %(times_loop_end)s
        """

    if configuration["hpc_profiling"]:
        wrapper += """
        %(timer_stop)s
        %(timer_end)s
        """

    wrapper += """
    }
    """

    return wrapper


def compose_openmp4gpu_wrapper(jitmodule):
    # if snapr_available():
    #     return snapr.optimize_code(jitmodule)
    wrapper = ""

    if configuration["hpc_save_result"]:
        wrapper += store_array % {"type": "int"}
        wrapper += store_array % {"type": "double"}

    wrapper += """
    double %(wrapper_name)s(int start, int end,
                      %(ssinds_arg)s
                      %(wrapper_args)s
                      %(const_args)s
                      %(layer_arg)s"""

    if configuration["hpc_profiling"]:
        wrapper += """
        %(other_args)s"""

    wrapper += ") {"

    wrapper += """
        %(user_code)s
        """
    if configuration["hpc_profiling"]:
        wrapper += """
        %(timer_declare)s
        %(timer_start)s
        """
    wrapper += """
        %(parallel_pragma_one)s
        {
        """
    if configuration["times"] > 1:
        wrapper += """
            %(times_loop_start)s
        """

    wrapper += """
            %(wrapper_decs)s;
            %(const_inits)s;
            %(map_decl)s
            %(vec_decs)s;
            %(parallel_pragma_two)s
            for ( int n = start; n < end; n++ ) {
                int i = %(index_expr)s;
                %(vec_inits)s;
                %(map_init)s;
                %(parallel_pragma_three)s
                %(extr_loop)s
                %(map_bcs_m)s;
                %(buffer_decl)s;
                %(buffer_gather)s

                %(kernel_name)s(%(kernel_args)s);

                %(itset_loop_body)s
                %(map_bcs_p)s;
                %(apply_offset)s;
                %(extr_loop_close)s
              }
              }
        """

    if configuration["times"] > 1:
        wrapper += """
        %(times_loop_end)s
        """

    if configuration["hpc_profiling"]:
        wrapper += """
        %(timer_stop)s
        %(timer_end)s
        """

    wrapper += """
    }
    """

    return wrapper
