import op2

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
    
    def handle_par_loop_call(self, kernel, it_space, *args):
        pass
    
    def handle_datacarrier_retrieve_value(self, datacarrier):
        assert isinstance(datacarrier, op2.DataCarrier)
        pass