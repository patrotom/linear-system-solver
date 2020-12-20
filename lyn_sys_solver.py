#!/usr/bin/env python3

import abc
import numpy as np
from numpy import linalg as la


class Solution():
    """
    The Solution class is a placeholder for the solution of the system of linear
    equations.
    """
    def __init__(self, found, it_num, xk):
        self.found = found
        self.it_num = it_num
        self.xk = xk


class LynSysSolver(abc.ABC):
    """
    The LynSysSolver class is an abstract class which contains the general
    behavior of the iterative methods for solving the linear systems.
    """
    def __init__(self, gamma, n, rel_prec, K):
        """
        * gamma - used as parameter for construction of the matrix A
        * n - size of the square matrix A
        * rel_prec - relative precision which is used to determine the validity
        of solution
        * K - maximum number of iterations
        """
        self.gamma = gamma
        self.n = n
        self.rel_prec = rel_prec
        self.K = K
        self.init_input()
        self.init_Q()

    def solve(self):
        """
        Solves the given linear system using an iterative method. If we do not
        find the solution after K iterations, we return the Solution object with
        the attribute found set to False.
        """
        prev_xk = self.x0

        for k in range(self.K):
            xk = self.calc_xk(prev_xk)
            rk = self.A * xk - self.b
            if self.check_rel_prec(rk):
                return Solution(True, k + 1, xk)

            prev_xk = xk.copy()
        
        return Solution(False, K, None)

    def init_input(self):
        """
        Initializes the matrix A, vector b, and initial vector x0.
        """
        m1 = np.diagflat([-1] * (self.n - 1), -1)
        m2 = np.diagflat([-1] * (self.n - 1), 1)
        m3 = np.diagflat([self.gamma] * self.n)
        self.A = np.matrix((m1 + m2 + m3).astype(np.double))

        self.b = np.matrix(
            np.full((self.n, 1), self.gamma - 2).astype(np.double)
        )
        self.b[0] = self.gamma - 1
        self.b[self.n - 1] = self.gamma - 1

        self.x0 = np.matrix(
            np.full((self.n, 1), 0).astype(np.double)
        )

    def norm(self, arr):
        """
        Calculates the maximum norm of a matrix or vector.
        """
        return la.norm(arr, np.inf)
    
    def calc_xk(self, prev_xk):
        """
        Calculates a current solution vector xk using the given formula:
        
        Q^(−1) * ((Q − A) * x_{k−1} + b)
        """
        return la.inv(self.Q) * ((self.Q - self.A) * prev_xk + self.b)

    def check_rel_prec(self, rk):
        """
        Checks the convergence criteria by comparing it with the specified
        relative precision.
        """
        return self.norm(rk) / self.norm(self.b) < self.rel_prec

    @abc.abstractmethod
    def init_Q(self):
        """
        Abstract method to define the Q matrix. Each algorithm (method) has to
        override this method and add its own implementation.
        """
        self.Q = None


class Jacobi(LynSysSolver):
    """
    Standard Jacobi method where matrix Q = D.
    """
    def init_Q(self):
        """
        Overrides the Q matrix init with the method specific implementation.
        """
        self.Q = np.matrix(np.diagflat(np.diag(self.A)))


class GaussSeidel(LynSysSolver):
    """
    Standard Gauss-Seidel method where matrix Q = D + L.
    """
    def init_Q(self):
        """
        Overrides the Q matrix init with the method specific implementation.
        """
        self.Q = np.matrix(np.tril(self.A))


if __name__ == "__main__":
    n = 20
    rel_prec = 10 ** -6
    K = 1000

    print("Jacobi method:\n")
    for gamma in range(3, 0, -1):
        sol = Jacobi(gamma, n, rel_prec, K).solve()
        print(f"Gamma = {gamma}, Found = {sol.found}, Iterations = {sol.it_num}")
    
    print("\n===============================\n")

    print("Gauss-Seidel method:\n")
    for gamma in range(3, 0, -1):
        sol = GaussSeidel(gamma, n, rel_prec, K).solve()
        print(f"Gamma = {gamma}, Found = {sol.found}, Iterations = {sol.it_num}")
