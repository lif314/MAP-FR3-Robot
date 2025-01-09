from spatialmath.base import r2x, q2r


q = [ 0.99993288,  0.00774572, -0.00537991,  0.00673031]

r = q2r(q, order='xyzs')


rpy = r2x(r, representation='rpy/xyz')

print("rpy: ", rpy) # rpy:  [-0.01556433 -0.01065504  3.12804841]

