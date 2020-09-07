"""
This file contains the implementation of 'Dissimilarity-based Sparse Subset Selection' algorithm using different
types of optimization techniques such as, message passing, greedy algorithm, and ADMM.
"""

import numpy as np
from numpy import linalg as LA
from Utils.ADMM import ADMM


#np.set_printoptions(threshold=np.nan)

class DS3(object):
    """
    :param dis_matrix:  dis-similarity matrix for the dataset calculated based on euclideon distance.
    :param reg:         regularization parameter

    """
    def __init__(self, dis_matrix, reg):
        self.reg = reg
        self.dis_matrix = dis_matrix
        self.N = len(self.dis_matrix)

    def regCost(self, z, p):
        """
        This function calculates the total cost of choosing the as few representatives as possible.

        :param z: matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param p: norm to be used to calculate regularization cost.

        :returns: regularization cost.
        """

        cost = 0
        for i in range(len(self.dis_matrix)):
            norm = LA.norm(z[i], ord=p)
            cost += norm

        return cost * self.reg

    def encodingCost(self, z):
        """
        This function calculates the total cost of encoding using all the representatives.

        :param z: matrix whose non-zero rows corresponds to the representatives of the dataset.

        :returns: encoding cost.
        """

        cost = 0
        for j in range(len(self.dis_matrix)):
            for i in range(len(self.dis_matrix)):
                try:
                    cost += self.dis_matrix[i, j] * z[i, j]
                except:
                    break

        return cost

    def transitionCost(self, z, M, m0):
        """
        This function calculates the total cost of transitions between the representatives.

        :param z:  matrix whose non-zero rows corresponds to the representatives of the dataset.
        :param M:  transition probability matrix for the states in the source set.
        :param m0: initial probability vector of the states in the source set.

        :returns: transition cost.
        """

        sum1 = 0
        for i in range(1, self.N):
            sum1 += np.matmul(np.matmul(np.transpose(z[:,(i-1)]), M), z[:, i])
        sum2 = np.matmul(z[:, 1], m0)

        return sum1 + sum2


    def ADMM(self, mu, epsilon, max_iter, p):
        """
        This function finds the subset of the data that can represent it as closely as possible given the
        regularization parameter. It uses 'alternating direction methods of multipliers' (ADMM) algorithm to
        solve the objective function for this problem, which is similar to the popular 'facility location problem'.

        To know more about this, please read :
        Dissimilarity-based Sparse Subset Selection
        by Ehsan Elhamifar, Guillermo Sapiro, and S. Shankar Sastry
        https://arxiv.org/pdf/1407.6810.pdf

        :param mu:        penalty parameter.
        :param epsilon:   small value to check for convergence.
        :param max_iter:  maximum number of iterations to run this algorithm.
        :param p:         norm to be used.

        :returns: representative of the data, total number of representatives, and the objective function value.
        """

        # initialize the ADMM class.
        G = ADMM(mu, epsilon, max_iter, self.reg)

        # run the ADMM algorithm.
        z_matrix = G.runADMM(self.dis_matrix, p)

        # new representative matrix obtained after changing largest value in each column to 1 and other values to 0.
        new_z_matrix = np.zeros(np.array(z_matrix).shape)
        idx = np.argmax(z_matrix, axis=0)
        
        for k in range(len(idx)):
            try:
                new_z_matrix[idx[k], k] = 1
            except:
                break

        # obj_func_value = self.encodingCost(z_matrix) + self.regCost(z_matrix, p)
        obj_func_value = self.encodingCost(z_matrix) + self.regCost(z_matrix, np.inf)

        # obj_func_value_post_proc = self.encodingCost(new_z_matrix) + self.regCost(new_z_matrix, p)
        obj_func_value_post_proc = self.encodingCost(new_z_matrix) + self.regCost(new_z_matrix, np.inf)

        # find the index and total count of the representatives, given the representative matrix.
        data_rep = []
        count = 0
        for i in range(len(z_matrix)):
            flag = 0
            for j in range(len(z_matrix)):
                try:
                    if z_matrix[i, j] > 0.1:
                        flag = 1
                        count += 1
                except:
                    break
            if flag == 1:
                data_rep.append(i)

        return data_rep, len(data_rep), obj_func_value, obj_func_value_post_proc, new_z_matrix
