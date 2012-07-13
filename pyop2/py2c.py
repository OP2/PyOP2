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
        func = self.parseFuncDef(node) + " {\n"

        for name, argtype in self.symtable.items():
            if any(map(lambda x: x.id == name, node.args.args)):
                continue
            func += argtype + " " + name + ";\n"

        func += "\n".join(map(self.visit, node.body))
        func += "}"
        return func

    def visit_Call(self, node):
        assert len(node.args) == 1
        if(node.func.id == "float"):
            return "(float)(" + self.visit(node.args[0]) + ")"
        elif(node.func.id == "int"):
            return "(int)(" + self.visit(node.args[0]) + ")"
        assert False

    def visit_If(self, node):
        assert isinstance(node, ast.If)

        code = ("if(" + self.visit(node.test) + ") {"
                + "\n".join(map(self.visit, node.body))
                + "}")

        if len(node.orelse) > 0:
            code += "else " + " else ".join(map(self.visit, node.orelse))

        return code

    def visit_Compare(self, node):
        assert isinstance(node, ast.Compare)
        assert len(node.comparators) == 1
        assert len(node.ops) == 1

        return (self.visit(node.left)
                + self.getOp(node.ops[0])
                + self.visit(node.comparators[0]));

    def visit_For(self, node):
        assert isinstance(node.iter, ast.Call)
        assert node.iter.func.id == "range" or node.iter.func.id == "xrange"

        args = node.iter.args
        assert len(args) > 0 and len(args) <= 3

        start = "0"
        step = "1"

        if(len(args) == 1):
            end = self.visit(args[0])
        else:
            end = self.visit(args[1])

        if(len(args) >= 2):
            start = self.visit(args[0])
        if(len(args) == 3):
            step = self.visit(args[2])

        var = node.target.id

        return ("for(" + var + " = " + start + ";"
                + step + " >= 0 ? " + var + " < " + end + " : " + var + " > " + end + ";"
                + var + " += " + step + ")"
                + " {"
                + "\n".join(map(self.visit, node.body))
                + "}")

    def visit_While(self, node):
        assert isinstance(node, ast.While)
        assert len(node.orelse) == 0
        return ("while(" + self.visit(node.test) + ") {"
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
        if(node.id == "True"):
            return "true"
        if(node.id == "False"):
            return "false"

        assert node.id in self.symtable
        return ("*" if self.symtable[node.id][-1] == "*" else "") + node.id

    def visit_Break(self, node):
        return "break;"

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
        if(isinstance(op, ast.Gt)):
            return ">"
        if(isinstance(op, ast.GtE)):
            return ">="
        if(isinstance(op, ast.Lt)):
            return "<"
        if(isinstance(op, ast.LtE)):
            return "<="
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

    def visit_If(self, node, symtable):
        assert isinstance(node, ast.If)
        assert len(self.visit(node.test, symtable)) > 0

        symtable = reduce(lambda symtable, node: self.visit(node, symtable), node.body, symtable)
        symtable = reduce(lambda symtable, node: self.visit(node, symtable), node.orelse, symtable)
        return symtable

    def visit_Compare(self, node, symtable):
        assert isinstance(node, ast.Compare)
        assert len(node.comparators) == 1

        left = self.visit(node.left, symtable)
        right = self.visit(node.comparators[0], symtable)
        assert len(intersect(left, right)) > 0
        return symtable

    def visit_For(self, node, symtable):
        assert isinstance(node, ast.For)
        assert isinstance(node.iter, ast.Call)
        assert isinstance(node.iter.func, ast.Name)
        assert node.iter.func.id == "range" or node.iter.func.id == "xrange"
        assert len(node.orelse) == 0
        assert isinstance(node.target, ast.Name)

        symtable[node.target.id] = ["int"]
        map(lambda x: self.visit(x, symtable), node.iter.args)
        return reduce(lambda symtable, node: self.visit(node, symtable), node.body, symtable)

    def visit_While(self, node, symtable):
        assert isinstance(node, ast.While)
        assert len(node.orelse) == 0
        self.visit(node.test, symtable)
        return reduce(lambda symtable, node: self.visit(node, symtable), node.body, symtable)

    def visit_Subscript(self, node, symtable):
        assert isinstance(node, ast.Subscript)
        assert isinstance(node.slice, ast.Index)
        assert isinstance(node.value, ast.Name)

        var = node.value.id
        assert var in symtable

        return map(lambda x: x[0:-2], intersect(symtable[var], self.arraytypes))

    def visit_Call(self, node, symtable):
        assert isinstance(node, ast.Call)
        assert isinstance(node.func, ast.Name)

        if node.func.id == "float":
            return ["float"]
        if node.func.id == "int":
            return ["int"]
        assert False, "Unknown function: " + node.func.id

    def visit_Return(self, node, symtable):
        assert isinstance(node, ast.Return)
        assert isinstance(node.value, ast.Name)
        assert node.value.id == "None"
        return symtable

    def visit_Break(self, node, symtable):
        assert isinstance(node, ast.Break)
        return symtable

    def visit_Name(self, node, symtable):
        assert isinstance(node, ast.Name)
        assert node.id in symtable or node.id == "True" or node.id == "False"

        if node.id == "True" or node.id == "False":
            return ["int"]

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
