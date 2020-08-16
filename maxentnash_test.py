import pytest
import numpy as np
from maxentnash import solve


eps_values = list(np.geomspace(1.0e-12, 1.0e-6, num=5))


class TestMaxentnashSolve:
    @pytest.mark.parametrize("eps", eps_values)
    def test_example1(self, eps):
        A = np.array([[0.0, 4.6, -4.6],
                      [-4.6, 0.0, 4.6],
                      [4.6, -4.6, 0.0]])
        B = np.array([1.0/3.0, 1.0/3.0, 1.0/3.0])
        assert(np.all(np.isclose(solve(A, eps=eps), B)))

    @pytest.mark.parametrize("eps", eps_values)
    def test_example2(self, eps):
        A = np.array([[0.0, 4.6, -4.6, -4.6],
                      [-4.6, 0.0, 4.6, 4.6],
                      [4.6, -4.6, 0.0, 0.0],
                      [4.6, -4.6, 0.0, 0.0]])
        B = np.array([1.0/3.0, 1.0/3.0, 1.0/6.0, 1.0/6.0])
        assert(np.all(np.isclose(solve(A, eps=eps), B)))

    @pytest.mark.parametrize("eps", eps_values)
    def test_example3(self, eps):
        A = np.array([[1.0, 1.0, 0.0],
                      [0.0, 0.0, 0.0]])
        B = solve(A, eps=eps)
        assert(B.shape == (2,))
        assert(B[0] > B[1])
