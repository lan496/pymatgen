# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.

import unittest

import numpy as np
from pymatgen.analysis.smith_normal_form import SNF


class TestSNF(unittest.TestCase):

    def setUp(self):
        self.random_state = 0

    def test(self):
        np.random.seed(self.random_state)
        # test for square and non-square matrices
        list_size = [
            (100, 3, 3),
            (100, 7, 5),
            (100, 11, 13)
        ]

        for size in list_size:
            X = np.random.randint(-1, 1, size=size)
            for i in range(size[0]):
                D, L, R = SNF(X[i]).get_smith_normal_form()
                self.verify_snf(X[i], D, L, R)

    def verify_snf(self, M, D, L, R):
        # check decomposition
        D_re = np.dot(L, np.dot(M, R))
        self.assertTrue(np.array_equal(D_re, D))

        # check if D is diagonal
        D_diag = np.diagonal(D)
        rank = np.count_nonzero(D_diag)
        self.assertEqual(np.count_nonzero(D) - rank, 0)

        # check if D[i + 1, i + 1] divide D[i, i]
        for i in range(rank - 1):
            self.assertTrue(D_diag[i + 1] % D_diag[i] == 0)

        # check L and R are unimodular
        self.is_unimodular(L)
        self.is_unimodular(R)

    def is_unimodular(self, A):
        self.assertAlmostEqual(np.abs(np.linalg.det(A)), 1)

        A_inv = np.around(np.linalg.inv(A))
        self.assertTrue(np.allclose(np.eye(A.shape[0]), np.dot(A, A_inv)))


if __name__ == '__main__':
    unittest.main()
