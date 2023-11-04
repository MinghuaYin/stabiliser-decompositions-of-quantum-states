import cvxpy as cp
import numpy as np

# Fix random number generator so we can repeat the experiment.
np.random.seed(1)

# The threshold value below which we consider an element to be zero.

A = [[-1, -1, 0.707106781, 0, 0, 0], [-1, 1, 0, 0.707106781, 0, 0],
     [-1, -1j, 0, 0, 0.707106781, 0], [-1, 1j, 0, 0, 0, 0.707106781]]

b = [1, 0.707106781 + 0.707106781j, 0, 0, 0, 0]

# Create variable.
x_l1 = cp.Variable(4)
ll = 1

A_real = []
A_im = []
for v in A:
    A_real.append(np.real(v))
    A_im.append(np.imag(v))


# Create constraint.
#constraints = [A@x_l1 - b]
#constraints +=[A@x_l1 +b>= -0.001]

constraints = []

obj = cp.Minimize(cp.norm(A_real@x_l1 + A_im@x_l1 - b, 2) +
                  ll*(cp.norm(x_l1, 1)))

#prob = cp.Problem(obj, constraints)
prob = cp.Problem(obj)
solution = prob.solve()
print("status: {}".format(prob.status))

#nnz_l1 = (np.absolute(x_l1.value) > delta).sum()
print("optimal objective value: {}".format(obj.value))
print(x_l1.value)
