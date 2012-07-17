from pyop2 import op2, py2c
op2.init(backend='opencl', diags=0)

@py2c.pykernel(["q", "qold"],
               {"q, qold" : "double *",
                "n" : "uint"})
def save_soln(q, qold):
    for n in range(4):
        qold[n] = q[n]

@py2c.pykernel(["x", "q", "adt"],
               {"x" : "double**",
                "q, adt" : "double*",
                "ri, dx, dy, u, v, c" : "double"})
def adt_calc(x, q, adt):
    ri = 1.0/q[0]
    u  = ri*q[1]
    v  = ri*q[2]
    c = sqrt(gam*gm1*(ri*q[3]-0.5*(u*u+v*v)))

    dx = x[1][0] - x[0][0]
    dy = x[1][1] - x[0][1]
    adt[0] = fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy)

    dx = x[2][0] - x[1][0]
    dy = x[2][1] - x[1][1]
    adt[0] += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy)

    dx = x[3][0] - x[2][0]
    dy = x[3][1] - x[2][1]
    adt[0] += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy)

    dx = x[0][0] - x[3][0]
    dy = x[0][1] - x[3][1]
    adt[0] += fabs(u*dy-v*dx) + c*sqrt(dx*dx+dy*dy)

    adt[0] = adt[0] / cfl

@py2c.pykernel(["x", "q", "adt", "res"],
               {"x, q, adt, res" : "double**",
                "dx, dy, mu, ri, p1, vol1, p2, vol2, f" : "double"})
def res_calc(x, q, adt, res):
    dx = x[0][0] - x[1][0]
    dy = x[0][1] - x[1][1]

    ri = 1.0/q[0][0]
    p1 = gm1*(q[0][3]-0.5*ri*(q[0][1]*q[0][1]+q[0][2]*q[0][2]))
    vol1 = ri*(q[0][1]*dy - q[0][2]*dx)

    ri = 1.0/q[1][0]
    p2 = gm1*(q[1][3]-0.5*ri*(q[1][1]*q[1][1]+q[1][2]*q[1][2]))
    vol2 = ri*(q[1][1]*dy - q[1][2]*dx)

    mu = 0.5*(adt[0]+adt[1])*eps

    f = 0.5*(vol1*q[0][1] + p1*dy + vol2*q[1][1] + p2*dy) + mu*(q[0][1]-q[1][1])
    res[0][1] += f
    res[1][1] -= f

    f = 0.5*(vol1*q[0][2] - p1*dx + vol2*q[1][2] - p2*dx) + mu*(q[0][2]-q[1][2])
    res[0][2] += f
    res[1][2] -= f

    f = 0.5*(vol1*(q[0][3]+p1) + vol2*(q[1][3]+p2)) + mu*(q[0][3]-q[1][3])
    res[0][3] += f
    res[1][3] -= f

@py2c.pykernel(["x", "q", "adt", "res", "bound"],
               {"x" : "double**",
                "q, adt, res" : "double*",
                "bound" : "int*",
                "dx, dy, mu, ri, p1, vol1, p2, vol2, f" : "double"})
def bres_calc(x, q, adt, res, bound):

    dx = x[0][0] - x[1][0]
    dy = x[0][1] - x[1][1]

    ri = 1.0/q[0]
    p1 = gm1*(q[3]-0.5*ri*(q[1]*q[1]+q[2]*q[2]))

    if bound[0] is 1:
        res[1] += p1*dy
        res[2] += p1*dx
    else:
        vol1 = ri*(q[1]*dy - q[2]*dx)

        ri = 1.0/qinf[0]
        p2 = gm1*(qinf[3]-0.5*ri*(qinf[1]*qinf[1]+qinf[2]*qinf[2]))
        vol2 = ri*(qinf[1]*dy - qinf[2]*dx)

        mu = adt[0]*eps

        f = 0.5*(vol1*q[0] + vol2*qinf[0]) + mu*(q[0]-qinf[0])
        res[0] += f
        f = 0.5*(vol1*q[1] + p1*dy + vol2*qinf[1] + p2*dy) + mu*(q[1]-qinf[1])
        res[1] += f
        f = 0.5*(vol1*q[2] - p1*dx + vol2*qinf[0] - p2*dx) + mu*(q[2]-qinf[2])
        res[2] += f
        f = 0.5*(vol1*(q[3]+p1) + vol2*(qinf[3]+p2)) + mu*(q[3]-qinf[3])
        res[3] += f

@py2c.pykernel(["qold", "q", "res", "adt", "rms"],
               {"qold, q, res, adt, rms" : "double*",
                "de, adti" : "double",
                "n" : "uint"})
def update(qold, q, res, adt, rms):

    adti = 1.0/adt[0]

    for n in range(4):
        de = adti*res[n]
        q[n] = qold[n] - de
        res[n] = 0.0
        rms[0] += de*de

print py2c.convert(save_soln)
print py2c.convert(adt_calc)
print py2c.convert(bres_calc)
print py2c.convert(update)
