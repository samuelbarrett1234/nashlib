import numpy as np
from scipy.optimize import (LinearConstraint, minimize)
from scipy.stats import entropy


"""
Author: Samuel Barrett
File description: Contains a `solve` function which finds the maximum
                  entropy Nash equilibrium for the row player. All
                  other functions in this file are internal helper
                  functions.

Relevant literature:

"Re-evaluating evaluation"
(https://arxiv.org/pdf/1806.02643.pdf)

"Maximum entropy correlated equilibria"
(http://proceedings.mlr.press/v2/ortiz07a/ortiz07a.pdf)
"""


def reshape_vars(m, n, P):
    """Reshape the vector P into its true matrix form. This is needed
    because scipy's `minimize` operates on vectors, even though our
    "variable" is naturally a matrix.

    Args:
        m: Number of rows
        n: Number of columns
        P: Vector of size m*n

    Returns:
        P reshaped into a matrix with m rows and n columns
    """
    assert(len(P.shape) == 1)
    assert(P.shape == (m * n,))
    return P.reshape((m, n))


def objective(m, n, P):
    """The objective function is just the negative entropy of the
    distribution of actions of the row player, based on the joint
    distribution of the players P. (This is why we sum across each
    row.)

    Args:
        m: The number of rows
        n: The number of columns
        P: The joint distribution (a vector of size m*n)

    Returns:
        The negative entropy of the row player's action distribution
        after marginalising-out the column player's actions.
    """
    P = reshape_vars(m, n, P)
    row_sums = np.sum(P, axis=1)
    assert(len(row_sums) == m)
    return -entropy(row_sums)


def grad_objective(m, n, P):
    """The gradient of the objective function.

    Args:
        m: The number of rows
        n: The number of columns
        P: The joint distribution (a vector of size m*n)

    Returns:
        A vector of size m*n which correspond to the partial
        derivatives of the objective function above.
    """
    P = reshape_vars(m, n, P)
    row_sums = np.sum(P, axis=1)
    grad = np.ones((m,)) + np.log(row_sums)
    grad = np.tile(grad, (n, 1)).T
    assert(grad.shape == (m, n))
    return grad.reshape((m * n,))


def get_payoff_constraint_functional(m, n, A, i1, i2, transpose=False):
    """Get the linear functional (represented by a vector)
    corresponding to the constraint on the players' actions
    which enforces that they are "rational". More specifically, ensure
    that the row player cannot expect to gain when switching from
    action i1 to action i2 unilaterally.

    Args:
        m: The number of rows
        n: The number of columns
        A: an m by n payoff matrix (where positive entries represent
           wins for the row player)
        i1: The index of the first action (switching *from*)
        i2: The index of the second action (switching *to*)
        transpose (bool, optional): If set to true, return the
                                    constraint for the column player
                                    instead of the row player.

    Returns:
        Returns a vector of size m*n such that the dot product of
        this vector with a probability distribution vector P, results
        in the expected gain if the row/col player unilaterally
        switches from action i1 to i2. Of course, this must be non-
        positive, because otherwise P would not be optimal (the player
        would have an incentive to switch.)
    """
    assert(A.shape == (m, n))

    # easier to conceptualise when written as a matrix first, then
    # flattened to a vector
    G = np.zeros((m, n))

    if not transpose:
        assert(i1 >= 0 and i2 >= 0 and i1 < m and i2 < m)
        for j in range(n):
            G[i1, j] = A[i2, j] - A[i1, j]
    else:
        assert(i1 >= 0 and i2 >= 0 and i1 < n and i2 < n)
        for j in range(m):
            G[j, i1] = A[j, i2] - A[j, i1]
        G *= -1  # swap scores around for col player

    return G.reshape((m * n,))


def get_probability_functional(m, n):
    """Get a linear functional (represented as a vector) which enforces
    that the entries of P sum to 1, as they are a probability
    distribution.

    Args:
        m: number of rows
        n: number of columns

    Returns:
        [type]: [description]
    """
    return np.ones((m * n,))


def get_bounds(m, n):
    """Probabilities must be in the range [0, 1] so we can supply scipy's
    optimiser with these box bounds on the distribution P. There is a
    separate constraint ensuring that the probabilities sum to 1.

    Args:
        m: The number of rows
        n: The number of columns

    Returns:
        A list of box bound tuples.
    """
    return [(0.0, 1.0) for i in range(m * n)]


def get_linear_constraints(m, n, A, eps):
    """Get the list of linear constraint objects.

    Args:
        m: The number of rows
        n: The number of columns
        A: A payoff matrix of size m by n. Positive entries represent
           wins for the row player.
        eps: The "rationality slack". If set to 0, the players are
             maximally rational, however the feasible set will be
             smaller so the optimiser might have a hard time finding
             a solution.

    Returns:
        A list of LinearConstraint objects.
    """
    assert(A.shape == (m, n))
    assert(eps >= 0.0)

    constr = []

    # constraints of rationality on the row player
    for i1 in range(m):
        for i2 in range(m):
            c = LinearConstraint(
                A=get_payoff_constraint_functional(m, n, A, i1, i2),
                lb=-np.inf,
                ub=eps)
            constr.append(c)

    # constraints of rationality on the col player
    for j1 in range(n):
        for j2 in range(n):
            c = LinearConstraint(
                A=get_payoff_constraint_functional(m, n, A, j1, j2,
                                                   transpose=True),
                lb=-np.inf,
                ub=eps)
            constr.append(c)

    # probabilities sum to one
    constr.append(LinearConstraint(A=get_probability_functional(m, n),
                                   lb=1.0, ub=1.0))

    return constr


def solve(A, eps=1.0e-6):
    """Given a payoff matrix, solve it to find the maximum entropy
    Nash equilibrium.

    Args:
        A: A payoff matrix of size m by n. Positive entries represent
           wins for the row player.
        eps: The "rationality slack". If set to 0, the players are
             maximally rational, however the feasible set will be
             smaller so the optimiser might have a hard time finding
             a solution.

    Returns:
        None if optimisation procedure failed, otherwise returns a
        vector of length A.shape[0] which is the Nash distribution
        over the row player's actions.
    """
    m, n = A.shape
    result = minimize(fun=lambda x: objective(m, n, x),
                      x0=np.ones((m * n,)) / (m * n),
                      jac=lambda x: grad_objective(m, n, x),
                      bounds=get_bounds(m, n),
                      constraints=get_linear_constraints(m, n, A, eps=eps))

    if not result.success:
        return None
    else:
        return np.sum(result.x.reshape(m, n), axis=1)  # marginalise out cols
