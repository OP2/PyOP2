import backend
import pyopencl as cl
import op2

class OpenCL(backend.Backend):
    """
    Checks for valid usage of the interface,
    but actually does nothing
    """
    def __init__(self):
        self._ctx = cl.create_some_context()
        self._queue = cl.CommandQueue(self._ctx)
        
        # Runtime parameter
        # TODO: actually load from the opencl platform/context/device information
        self._warpsize = 1
        self._threads_per_block = self._ctx.get_info(cl.context_info.DEVICES)[0].get_info(cl.device_info.MAX_WORK_GROUP_SIZE)
        self._blocks_per_grid = 2
        self._thread_count = self._threads_per_block * self._blocks_per_grid
        
    
        # Load string templates
        self._direct_loop_stg = stringtemplate3.StringTemplateGroup(file=stringtemplate3.StringIO(template_group_direct), lexer="default")
    
    def handle_kernel_declaration(self, kernel):
        assert isinstance(kernel, op2.Kernel)
        # don t need to do anything yet
        pass
    
    def handle_datacarrier_declaration(self, datacarrier):
        assert isinstance(datacarrier, op2.DataCarrier)
        if (isinstance(datacarrier, op2.Dat):
            buf = cl.Buffer(self._ctx, cl.mem_flags.READ_WRITE, size=datacarrier._data.nbytes)
            cl.enqueue_write_buffer(self._queue, buf, datacarrier._data).wait()
            self._buffers[datacarrier] = buf
        else:
            raise NotImplemented()
        pass
    
    def handle_map_declaration(self, map):
        assert isinstance(map, op2.Map)
        # dirty how to do that properly ?
        if not map._name == 'identity':
            buf = cl.Buffer(self._ctx, cl.mem_flags.READ_ONLY, size=map._values.nbytes)
            cl.enqueue_write_buffer(self._queue, buf, map._values).wait()
            self._buffers[map] = buf
    
    def _handle_par_loop_call(self, parloop):
        if parloop.is_direct():
            # compute runtime params
            dynamic_shared_memory_size = max(map(lambda a: a['dat']._dim * a['dat']._datatype.nbytes, parloop._args))
            shared_memory_offset = dynamic_shared_memory_size * self._warpsize
            dynamic_shared_memory_size = dynamic_shared_memory_size * self._threads_per_block
            
            # codegen
            template = self._direct_loop_stg.getInstanceOf("direct_loop")
            template['parloop'] = parloop
            template['const'] = {"warpsize": self._warpsize,\
                                 "shared_memory_offset": shared_memory_offset,\
                                 "dynamic_shared_memory_size": dynamic_shared_memory_size}
            source = str(template)
            prg = cl.Program (self._ctx, source).build(options="-Werror")
            kernel = prg.__getattr__(parloop._kernel._name + '_stub')
            for i, a in parloop._args:
                kernel.set_arg(i, self._buffers[a['dat']])
            
            cl.enqueue_nd_range_kernel(self._queue, kernel, (self._thread_count,), (self._threads_per_block,), g_times_l=False).wait()
            
        else:
            raise NotImplementedError()

    def handle_datacarrier_retrieve_value(self, datacarrier):
        assert isinstance(datacarrier, op2.DataCarrier)
        pass

_template_direct = """
group opencl_direct;

direct_loop(parloop,const)::=<<
$header(const)$
$parloop._kernel._code$
$kernel_stub(parloop=parloop)$
>>

kernel_stub(parloop)::=<<
__kernel
void $parloop._kernel._name$_stub (
  $parloop._args:{__global $it._dat._cl_type$* $it._dat._name$};separator=",\n"$
)
{
  unsigned int shared_memory_offset = $const.shared_memory_offset$;
  unsigned int set_size = $parloop._it_space._size$;

  __local char shared[$const.dynamic_shared_memory_size$];
  __local char* shared_pointer;
  
  $parloop._stagged_args:{__private $it._dat._cl_type$ $it._dat._name$_local[$it._dat._dim$];};separator="\n"$

  int i_1;
  int i_2;

  int local_offset;
  int active_threads_count;
  int thread_id;

  thread_id = get_local_id(0) % OP_WARPSIZE;
  shared_pointer = shared + shared_memory_offset * (get_local_id(0) / OP_WARPSIZE);

  for (i_1 = get_global_id(0); i_1 < set_size; i_1 += get_global_size(0))
  {
    local_offset = i_1 - thread_id;
    active_threads_count = MIN(OP_WARPSIZE, set_size - local_offset);
  
    $parloop._stagged_in_args:stagein();separator="\n"$
    $kernel_call(parloop=parloop)$
    $parloop._stagged_out_args:stageout();separator="\n"$
  }
}
>>

stagein(arg)::=<<
// $arg._dat._name$
for (i_2 = 0; i_2 < $arg._dat._dim$; ++i_2) {
  (($arg._dat._cl_type$*) shared_pointer)[thread_id + i_2 * active_threads_count] = $arg._dat._name$[thread_id + i_2 * active_threads_count + local_offset * 1];
}
for (i_2 = 0; i_2 < $arg._dat._dim$; ++i_2) {
  $arg._dat._name$_local[i_2] = (($arg._dat._cl_type$*) shared_pointer)[i_2 + thread_id * 1];
}
>>

stageout(arg)::=<<
// $arg._dat._name$
for (i_2 = 0; i_2 < $arg._dat._dim$; ++i_2) {
  (($arg._dat._cl_type$*) shared_pointer)[i_2 + thread_id * 1] = $arg._dat._name$_local[i_2];
}
for (i_2 = 0; i_2 < $arg._dat._dim$; ++i_2) {
  $arg._dat._name$[thread_id + i_2 * active_threads_count + local_offset * 1] = (($arg._dat._cl_type$*) shared_pointer)[thread_id + i_2 * active_threads_count];
}
>>
  
kernel_call(parloop)::=<<$parloop._kernel._name$($parloop._args:{$it._dat._name$_local};separator=", "$);>>

  
header(const)::=<<
#define OP_WARPSIZE $const.warpsize$
#define MIN(a,b) ((a < b) ? (a) : (b))
>>

"""