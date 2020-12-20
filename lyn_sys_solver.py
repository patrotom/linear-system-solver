#!/usr/bin/env python3

import numpy as np


class LynSysSolver():
    def __init__(self, gamma, n, rel_prec):
        self.gamma = gamma
        self.n = n
        self.rel_prec = rel_prec
        self.init_input()

    def init_input(self):
        m1 = np.diagflat([-1] * (self.n - 1), -1)
        m2 = np.diagflat([-1] * (self.n - 1), 1)
        m3 = np.diagflat([self.gamma] * self.n)
        self.mat_a = (m1 + m2 + m3).astype(np.double)

        self.vec_b = np.full((self.n, 1), self.gamma - 2).astype(np.double)
        self.vec_b[0] = self.gamma - 1
        self.vec_b[self.n - 1] = self.gamma - 1

    def pretty_print(self, arr):
        print("\n".join([" ".join([str(cell) for cell in row]) for row in arr]))


if __name__ == "__main__":
    n, rel_prec = 20, 10 ** -6
    sol = LynSysSolver(3, n, rel_prec)

