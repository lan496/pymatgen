# coding: utf-8
# Copyright (c) Pymatgen Development Team.
# Distributed under the terms of the MIT License.


__author__ = "Kohei Shinohara"
__copyright__ = "Copyright 2019, The Materials Project"
__version__ = "0.1"
__maintainer__ = "Kohei Shinohara"
__date__ = "Feb 20, 2019"


"""
This module contains an algorithm for caculating Smith normal form.
"""
from copy import deepcopy

import numpy as np


class SNF:
    """
    This class caculates Smith normal form of a given matrix
    by extended Euclidean algorithm.

    see the following post for a detailed description of this algorithm:
        http://www.dlfer.xyz/post/2016-10-27-smith-normal-form/

    Parameters
    ----------
    A: array, (m, n)
        interger matrix
    """
    def __init__(self, A):
        self.A_org = A
        self.A_ = deepcopy(self.A_org)
        self.left = np.eye(self.A_.shape[0], dtype=int)
        self.right = np.eye(self.A_.shape[1], dtype=int)

    @property
    def num_row(self):
        return self.A_.shape[0]

    @property
    def num_column(self):
        return self.A_.shape[1]

    def get_smith_normal_form(self):
        """
        calculate Smith normal form

        Returns
        -------
        D: array, (m, n)
        left: array, (m, m)
        right: array, (n, n)
            D = np.dot(left, np.dot(M, right))
            left, right are unimodular.
        """
        D, left, right = self._snf(s=0)
        return D, left, right

    def _snf(self, s):
        """
        determine SNF up to the s-th row and column elements
        """
        if s == min(self.A_.shape):
            return self.A_, self.left, self.right

        # choose a pivot
        row, col = get_nonzero_min_abs(self.A_, s)
        if col is None:
            # if there does not remain non-zero elements, this procesure ends.
            return self.A_, self.left, self.right
        self._swap_columns(s, row)
        self._swap_rows(s, col)

        # eliminate the s-th column entries
        for i in range(s + 1, self.num_row):
            if self.A_[i, s] != 0:
                k = self.A_[i, s] // self.A_[s, s]
                self._add_column(i, s, -k)

        # eliminate the s-th row entries
        for j in range(s + 1, self.num_column):
            if self.A_[s, j] != 0:
                k = self.A_[s, j] // self.A_[s, s]
                self._add_row(j, s, -k)

        # if there does not remain non-zero element in s-th row and column, find a next entry
        if (np.count_nonzero(self.A_[s, (s + 1):]) == 0) \
                and (np.count_nonzero(self.A_[(s + 1):, s]) == 0):
            row_next, _ = self._find_non_diviable_element(s)
            if row_next is not None:
                # move non-diviable elements by A[s, s] into s-th column
                self._add_column(s, row_next, 1)
                return self._snf(s)
            elif self.A_[s, s] < 0:
                self._change_sign_column(s)
            return self._snf(s + 1)
        else:
            return self._snf(s)

    def _find_non_diviable_element(self, s):
        """
        return entry which is not diviable by A[s, s]
        assume A[s, s] is not zero.
        """
        for i in range(s + 1, self.num_row):
            for j in range(s + 1, self.num_column):
                if self.A_[i, j] % self.A_[s, s] != 0:
                    return i, j
        return None, None

    def _swap_columns(self, axis1, axis2):
        self.left[[axis1, axis2]] = self.left[[axis2, axis1]]
        self.A_[[axis1, axis2]] = self.A_[[axis2, axis1]]

    def _swap_rows(self, axis1, axis2):
        self.right[:, [axis1, axis2]] = self.right[:, [axis2, axis1]]
        self.A_[:, [axis1, axis2]] = self.A_[:, [axis2, axis1]]

    def _change_sign_column(self, axis):
        self.left[axis] *= -1
        self.A_[axis] *= -1

    def _change_sign_row(self, axis):
        self.right[:, axis] *= -1
        self.A_[:, axis] *= -1

    def _add_column(self, axis1, axis2, k):
        """
        add k times axis2 to axis1
        """
        self.left[axis1] += self.left[axis2] * k
        self.A_[axis1] += self.A_[axis2] * k

    def _add_row(self, axis1, axis2, k):
        """
        add k times axis2 to axis1
        """
        self.right[:, axis1] += self.right[:, axis2] * k
        self.A_[:, axis1] += self.A_[:, axis2] * k


class SupercellHash:
    """
    perfect hash function for checking if two lattice points are equivalent up to supercell translation.
    see the detailed in Gus L. W. Hart and Rodney W. Forcade,
    "Algorithm for generating derivative structures," Phys. Rev. B 77 224115, (26 June 2008)

    Parameters
    ----------
    scale_matrix: array, (3, 3)
    """

    def __init__(self, scale_matrix):
        self.scale_matrix = scale_matrix

        D, left, right = SNF(self.scale_matrix).get_smith_normal_form()
        self.D = D
        self.left = left
        self.right = right
        self.right_inv = np.linalg.inv(self.right)

    @property
    def shape(self):
        return tuple(self.D.diagonal())

    def hash_image(self, indexes, return_supercell_jimage=False):
        """
        hash lattice point

        Parameters
        ----------
        indexes: array
        return_supercell_jimage: if True, return jimage in supercell

        Returns
        -------
        remainder: array
            It it equivalent to be symmetrically equal up to supercell translation
            to match remainder for two lattice point
        (Optional) supercell_jiamge: array
        """
        factor = np.dot(self.right.T, np.array(indexes))
        remainder = np.around(np.mod(factor, np.array(self.shape))).astype(int)
        if return_supercell_jimage:
            supercell_jimage = (np.dot(self.right.T, indexes) - remainder) / self.D.diagonal()
            supercell_jimage = np.dot(self.left.T, supercell_jimage)
            supercell_jimage = np.around(supercell_jimage).astype(int)

            return remainder, supercell_jimage
        else:
            return remainder

    def unhash_factor(self, factor):
        indexes = np.dot(self.right_inv.T, np.array(factor))
        return indexes


def get_nonzero_min_abs(A, s):
    """
    return idx = argmin_{i, j} abs(A[i, j]) s.t. (i >= s and j >= s and A[i, j] != 0)
    if failed, return (None, None)
    """
    idx = (None, None)
    valmin = None

    for i in range(s, A.shape[0]):
        for j in range(s, A.shape[1]):
            if A[i, j] == 0:
                continue
            if (valmin is None) or (np.abs(A[i, j]) < valmin):
                idx = (i, j)
                valmin = np.abs(A[i, j])
    return idx
