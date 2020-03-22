import numpy as np
from scipy.optimize import linprog


"""Find any Nash equilibrium for the following two
player game. The payoff matrix should have shape (n, m)
where n is the number of rows and m the number of columns
corresponding to the actions of the maximising player and
the minimising player, respectively.
"""
def find_nash(payoff_mat):
    assert(isinstance(payoff_mat, np.ndarray))
    assert(len(payoff_mat.shape) == 2)
    n, m = payoff_mat.shape

    # make the payoff matrix matrix have strictly positive entries:
    # (this does not change the solution set)
    A = payoff_mat - payoff_mat.min() + 1.0

    # set up the linear program:
    # we will add an extra variable z and do the following:
    # minimise -z (note: scipy does min by default)
    # subject to:
    # for each 1 <= j <= m, sum_{i=1}^n p_i A_{i,j} >= z
    # sum_{i=1}^n p_i <= 1
    # z, p_1, ..., p_n >= 0

    c = np.array([1.0] + [0.0] * n)  # objective coefficients

    p_constr = np.array([0.0] + [1.0] * n)  # sum of probabilities <= 1

    # value constraints
    A_constr = np.concatenate((np.ones((m, 1)), -A.T), axis=1)
    b_constr = np.zeros((m,))

    # now combine the constraints
    C = np.concatenate((A_constr, p_constr.reshape((1, -1))), axis=0)
    b = np.concatenate((b_constr, [1.0]))

    result = linprog(-c,
                     C, b,
                     bounds=[(0.0, None)] + [(0.0, 1.0)] * n
                     )
    
    # now examine the returned result:
    if not result.success:
        return None
    else:
        return result.x[1:]
