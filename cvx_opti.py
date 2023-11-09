import math
import cvxpy as cp
import numpy as np

B = np.array([[1, 1, -1.414213562, 0, 0, 0],
              [1, -1, 0, -1.414213562, 0, 0],
              [1, 1j, 0, 0, -1.414213562, 0],
              [1, -1j, 0, 0, 0, -1.414213562]]).T

c = np.array([0.707106781, 0.5 + 0.5j, 0, 0, 0, 0])

x_l1 = cp.Variable(4, complex=True)

obj = cp.Minimize(cp.norm(B@x_l1 + c, 1)**2)

prob = cp.Problem(obj)
solution = prob.solve()
print(f"status: {prob.status}")

print(f"optimal objective value: {obj.value}")
print(repr(x_l1.value))
print(f'{4/(2 + math.sqrt(2)) = }')
