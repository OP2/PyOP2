import op2

class ParLoopCall(object):
    """
    Backend Agnostic support code
    """
    def __init__(self, kernel, it_space, *args):
        assert ParLoopCall.check(kernel, it_space, *args)
        self._kernel = kernel
        self._it_space = it_space
        self._args = *args
        pass

    def check(kernel, it_space, *args):
        return false

class Backend(object):
    """
    Generic backend interface
    """
    def __init__(self):
        raise NotImplemented()

    def handle_kernel_declaration(self, kernel):
        raise NotIMplemented()
    
    def handle_datacarrier_declaration(self, datacarrier):
        raise NotImplemented()
    
    def handle_map_declaration(self, map):
        raise NotImplemented()    

    def handle_par_loop_call(self, kernel, it_space, *args):
        self._handle_par_loop_call(ParLoopCall(kernel, it_space, *args)

    def _handle_par_loop_call(self, parloop):
        raise NotImplemented()

    def handle_datacarrier_retrieve_value(self, datacarrier):
        raise NotImplemented()

class VoidBackend(Backend):
    """
    Checks for valid usage of the interface,
    but actually does nothing
    """
    def __init__(self):
        pass
    
    def handle_kernel_declaration(self, kernel):
        assert isinstance(kernel, op2.Kernel)
        pass
    
    def handle_datacarrier_declaration(self, datacarrier):
        assert isinstance(datacarrier, op2.DataCarrier)
        pass
    
    def handle_map_declaration(self, map):
        assert isinstance(map, op2.Map)
        pass
    
    def _handle_par_loop_call(self, parloop):
        pass
    
    def handle_datacarrier_retrieve_value(self, datacarrier):
        assert isinstance(datacarrier, op2.DataCarrier)
        pass