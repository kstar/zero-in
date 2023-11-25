import math
from math import cos, sin
import numpy as np
from pyquaternion import Quaternion

# Tests the following derivation
#
# Let q := cos(a) cos(b) + i cos(a) sin(b) + j sin(a) cos(c) + k sin(a) sin(c)
# (Generic representation for a unit quaternion)
#
# Then,
#
# q i q¯¹ = i cos(2a) + j sin(2a) sin(b + c) - k sin(2a) cos(b + c)
#
# q j q¯¹ = i sin(2a) sin(b - c) + j [ cos²(a) cos(2b) + sin²(a) cos(2c) ] + k [ cos²(a) sin(2b) + sin²(a) sin(2c) ]
#
# q k q¯¹ = i sin(2a) cos(b - c) + j [ sin²(a) sin(2c) - cos²(a) sin(2b) ] + k [ -sin²(a) cos(2c) + cos²(a) cos(2b) ]
#

L = np.linspace(-math.pi, math.pi, 21)
results_k, results_j, results_i = [], [], []
for a in L:
    for b in L:
        for c in L:
            q = Quaternion(cos(a) * cos(b), cos(a) * sin(b), sin(a) * cos(c), sin(a) * sin(c))
            results_i.append(q * Quaternion(0, 1, 0, 0) * q.inverse - Quaternion(
                0,
                cos(2 * a),
                sin(2 * a) * sin(b + c),
                -sin(2 * a) * cos(b + c),
            ))

            results_j.append(q * Quaternion(0, 0, 1, 0) * q.inverse - Quaternion(
                0,
                sin(2 * a) * sin(b - c),
                cos(a) * cos(a) * cos(2 * b) + sin(a) * sin(a) * cos(2 * c),
                cos(a) * cos(a) * sin(2 * b) + sin(a) * sin(a) * sin(2 * c),
            ))

            results_k.append(q * Quaternion(0, 0, 0, 1) * q.inverse - Quaternion(
                0,
                sin(2 * a) * cos(b - c),
                sin(a) * sin(a) * sin(2 * c) - cos(a) * cos(a) * sin(2 * b),
                -sin(a) * sin(a) * cos(2 * c) + cos(a) * cos(a) * cos(2 * b),
            ))
err_i, err_j, err_k = (np.max([ri.norm for ri in results_i]), np.max([rj.norm for rj in results_j]), np.max([rk.norm for rk in results_k]))

print(f'err_i = {err_i}, err_j = {err_j}, err_k = {err_k}')
assert err_i < 1e-10, err_i
assert err_j < 1e-10, err_j
assert err_k < 1e-10, err_k
