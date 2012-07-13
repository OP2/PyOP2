import meta.decompiler as meta
import meta.asttools as asttools
import ast

# A decorator for us to know kernel types

def kernel_types(types):
    def decorator(f):
        def __py2c_kernel__(*args):
            return (types, meta.decompile_func(f))
        return __py2c_kernel__
    return decorator

def convert(code):
    symbols = SymbolTableGenerator(code).generate()
    return PythonToCConverter(code, symbols).convert()

class PythonToCConverter(ast.NodeVisitor):
    def __init__(self, func, symbols):
        self.func = func
        self.symtable = symbols

    def parseFuncDef(self, funcDef):
        assert isinstance(funcDef, ast.FunctionDef)

        params = []
        for arg in funcDef.args.args:
            argType = self.symtable[arg.id]

            if(argType[-2:] == "[]"):
                argType = argType.replace("[]", "*")
            else:
                self.symtable[arg.id] = argType + "*"
                argType += "*"

            params += [[arg.id, argType]]


        func = "void " + funcDef.name + "("
        func += reduce(lambda acc, name: acc + (", " if acc != "" else "") + name[1] + " " + name[0], params, "")
        return func + ")"

    def convert(self):
        return self.visit(self.func()[1])

    def visit_FunctionDef(self, node):
        return (self.parseFuncDef(node)
                + " {"
                + "\n".join(map(self.visit, node.body))
                + "}")

    def visit_Call(self, node):
        assert len(node.args) == 1
        if(node.func.id == "float"):
            return "(float)(" + self.visit(node.args[0]) + ")"
        elif(node.func.id == "int"):
            return "(int)(" + self.visit(node.args[0]) + ")"
        assert False

    def visit_For(self, node):
        if(isinstance(node.iter, ast.Call) and node.iter.func.id == "range"):
            args = node.iter.args
            assert len(args) > 0 and len(args) <= 3

            start = 0
            step = 1

            if(len(args) == 1):
                end = args[0].n
            else:
                end = args[1].n

            if(len(args) >= 2):
                start = args[0].n
            if(len(args) == 3):
                step = args[2].n

            end = str(end)
            start = str(start)
            step = str(step)

            var = node.target.id
            varType = self.symtable[var]

            return ("for(" + varType + " " + var + " = " + start + ";"
                    + var + (" < " if int(step) >= 0 else " > ") + end + ";"
                    + var + " += " + step + ")"
                    + " {"
                    + "\n".join(map(self.visit, node.body))
                    + "}")

    def visit_Return(self, node):
        return "return " + (node.value.id if node.value.id != "None" else "") + ";"

    def visit_Assign(self, node):
        assert len(node.targets) == 1
        return self.visit(node.targets[0]) + " = " + self.visit(node.value) + ";"

    def visit_AugAssign(self, node):
        return self.visit(node.target) + self.getOp(node.op) + "= " + self.visit(node.value) + ";"

    def visit_Num(self, node):
        return str(node.n)

    def visit_Name(self, node):
        assert node.id in self.symtable
        return ("*" if self.symtable[node.id][-1] == "*" else "") + node.id

    def visit_BinOp(self, node):
        return self.visit(node.left) + self.getOp(node.op) + self.visit(node.right)

    def visit_Subscript(self, node):
        return self.visit(node.value) + "[" + self.visit(node.slice.value) + "]"

    def getOp(self, op):
        if(isinstance(op, ast.Add)):
            return "+"
        if(isinstance(op, ast.Sub)):
            return "-"
        if(isinstance(op, ast.Mult)):
            return "*"
        if(isinstance(op, ast.Div)):
            return "/"
        assert False

class SymbolTableGenerator(ast.NodeVisitor):
    def __init__(self, func):
        self.func = func
        self.basictypes = ["int", "float"]
        self.arraytypes = ["int[]", "float[]"]
        self.alltypes = self.basictypes + self.arraytypes

    def generate(self):
        assert self.func.__name__ == "__py2c_kernel__"
        types, ast = self.func()

        #TODO: Remove ugly hack when kernels automatically select the appropriate address space
        symtable = {}
        for name, origType in types.items():
            origType = origType.replace("__local ", "").replace("__global ", "").replace("__private ", "")
            if(origType[0] == "u"):
                symtable[name] = [origType, origType[1:]]
            else:
                symtable[name] = [origType]

        symtable = self.visit(ast, symtable)
        symbols = {}

        for name, symtypes in symtable.items():
            assert (len(intersect(self.basictypes, symtypes)) == 0
                 or len(intersect(self.arraytypes, symtypes)) == 0), "Unable to deduce variable type of " + name

            if "float" in symtypes:
                symbols[name] = "float"
            if "int" in symtypes:
                symbols[name] = "int"

            if name in types:
                symbols[name] = types[name]

        return symbols

    def visit(self, node, symtable):
        """Visit a node"""
        method = "visit_" + node.__class__.__name__
        visitor = getattr(self, method, self.generic_visit)
        return visitor(node, symtable)

    def generic_visit(self, node, symtable):
        """Unknown node type"""
        assert False, "Unknown node type: " + str(type(node))

    def visit_FunctionDef(self, node, symtable):
        # asttools.print_ast(node)
        assert isinstance(node, ast.FunctionDef)
        assert len(node.args.defaults) == 0
        assert node.args.kwarg is None
        assert node.args.vararg is None
        assert len(node.decorator_list) == 0

        for arg in node.args.args:
            assert isinstance(arg.ctx, ast.Param)
            if arg.id not in symtable:
                symtable[arg.id] = self.alltypes

        return reduce(lambda symtable, node: self.visit(node, symtable), node.body, symtable)

    def visit_AugAssign(self, node, symtable):
        assert isinstance(node, ast.AugAssign)
        assert isinstance(node.target, ast.Name) or isinstance(node.target, ast.Subscript)
        assert isinstance(node.target.ctx, ast.Store)

        if isinstance(node.target, ast.Name):
            assert node.target.id in symtable
            var = node.target.id
            symtable[var] = intersect(symtable[var], self.visit(node.value, symtable))
        else:
            assert isinstance(node.target.value, ast.Name)
            assert node.target.value.id in symtable
            var = node.target.value.id
            symtable[var] = intersect(symtable[var], map(lambda x: x + "[]", self.visit(node.value, symtable)))

        return symtable

    def visit_Assign(self, node, symtable):
        assert isinstance(node, ast.Assign)
        assert len(node.targets) == 1
        assert isinstance(node.targets[0], ast.Name)
        assert isinstance(node.targets[0].ctx, ast.Store)

        valueTypes = self.visit(node.value, symtable)

        var = node.targets[0].id

        if var not in symtable:
            symtable[var] = self.alltypes

        symtable[var] = intersect(symtable[var], valueTypes)
        assert len(symtable[var]) >= 1, "Could not deduce type of " + var
        return symtable

    def visit_Subscript(self, node, symtable):
        assert isinstance(node, ast.Subscript)
        assert isinstance(node.slice, ast.Index)
        assert isinstance(node.value, ast.Name)

        var = node.value.id
        assert var in symtable

        return map(lambda x: x[0:-2], intersect(symtable[var], self.arraytypes))

    def visit_Return(self, node, symtable):
        assert isinstance(node, ast.Return)
        assert isinstance(node.value, ast.Name)
        assert node.value.id == "None"
        return symtable

    def visit_Name(self, node, symtable):
        assert isinstance(node, ast.Name)
        assert node.id in symtable

        return symtable[node.id]

    def visit_Num(self, node, symtable):
        assert isinstance(node, ast.Num)

        types = []

        if isinstance(node.n, int):
            types += ["int", "float"]

        if isinstance(node.n, float):
            types += ["float"]

        return types

    def visit_BinOp(self, node, symtable):
        assert isinstance(node, ast.BinOp)

        left = self.visit(node.left, symtable)
        right = self.visit(node.right, symtable)

        return intersect(left, right)

def intersect(a, b):
    return list(set(a) & set(b))
