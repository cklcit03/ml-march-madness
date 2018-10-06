# Copyright (C) 2018  Caleb Lo
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <http://www.gnu.org/licenses/>.

# Methods for logistic regression
from scipy.optimize import fmin_ncg
import numpy


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def compute_sigmoid(z):
    """ Computes sigmoid function.
    Args:
      z: Can be a scalar, a vector or a matrix.
    Returns:
      sigmoid_z: Sigmoid function value.
    """
    sigmoid_z = 1/(1+numpy.exp(-z))
    return sigmoid_z


def compute_cost(theta, X, y, num_train_ex):
    """ Computes cost function J(\theta).
    Args:
      theta: Vector of parameters for logistic regression.
      X: Matrix of features.
      y: Vector of labels.
      num_train_ex: Number of training examples.
    Returns:
      j_theta: Logistic regression cost.
    Raises:
      An error occurs if the number of features is 0.
      An error occurs if the number of training examples is 0.
    """
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = X.shape[1]
    if num_features == 0: raise Error('num_features = 0')
    theta = numpy.reshape(theta, (num_features, 1), order='F')
    h_theta = compute_sigmoid(numpy.dot(X, theta))
    j_theta = (numpy.sum(numpy.subtract(numpy.multiply(-y, numpy.log(h_theta)),
                                        numpy.multiply((1-y),
                                                       numpy.log(1-h_theta))),
                         axis=0))/num_train_ex
    return j_theta


def compute_cost_reg(theta, X, y, num_train_ex, lamb):
    """ Computes regularized cost function J(\theta).
    Args:
      theta: Vector of parameters for regularized logistic regression.
      X: Matrix of features.
      y: Vector of labels.
      num_train_ex: Number of training examples.
      lamb: Regularization parameter.
    Returns:
      j_theta_reg: Regularized logistic regression cost.
    Raises:
      An error occurs if the number of features is 0.
      An error occurs if the number of training examples is 0.
    """
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = X.shape[1]
    if num_features == 0: raise Error('num_features = 0')
    theta = numpy.reshape(theta, (num_features, 1), order='F')
    h_theta = compute_sigmoid(numpy.dot(X, theta))
    theta_squared = numpy.power(theta, 2)
    j_theta = (numpy.sum(numpy.subtract(numpy.multiply(-y, numpy.log(h_theta)),
                                        numpy.multiply((1-y),
                                                       numpy.log(1-h_theta))),
                         axis=0))/num_train_ex
    j_theta_reg = (
        j_theta+(lamb/(2*num_train_ex))*numpy.sum(theta_squared,
                                                  axis=0)-theta_squared[0])
    return j_theta_reg


def compute_gradient(theta, X, y, num_train_ex):
    """ Computes gradient of cost function J(\theta).
    Args:
      theta: Vector of parameters for logistic regression.
      X: Matrix of features.
      y: Vector of labels.
      num_train_ex: Number of training examples.
    Returns:
      grad_array_flat: Vector of logistic regression gradients
                       (one per feature).
    Raises:
      An error occurs if the number of features is 0.
      An error occurs if the number of training examples is 0.
    """
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = X.shape[1]
    if num_features == 0: raise Error('num_features = 0')
    theta = numpy.reshape(theta, (num_features, 1), order='F')
    h_theta = compute_sigmoid(numpy.dot(X, theta))
    grad_array = numpy.zeros((num_features, 1))
    for grad_index in range(0, num_features):
        grad_term = numpy.multiply(numpy.reshape(X[:, grad_index],
                                                 (num_train_ex, 1)),
                                   numpy.subtract(h_theta, y))
        grad_array[grad_index] = (numpy.sum(grad_term, axis=0))/num_train_ex
    grad_array_flat = numpy.ndarray.flatten(grad_array)
    return grad_array_flat


def compute_gradient_reg(theta, X, y, num_train_ex, lamb):
    """ Computes gradient of regularized cost function J(\theta).
    Args:
      theta: Vector of parameters for regularized logistic regression.
      X: Matrix of features.
      y: Vector of labels.
      num_train_ex: Number of training examples.
      lamb: Regularization parameter.
    Returns:
      grad_array_reg_flat: Vector of regularized logistic regression gradients
                           (one per feature).
    Raises:
      An error occurs if the number of features is 0.
      An error occurs if the number of training examples is 0.
    """
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    num_features = X.shape[1]
    if num_features == 0: raise Error('num_features = 0')
    theta = numpy.reshape(theta, (num_features, 1), order='F')
    h_theta = compute_sigmoid(numpy.dot(X, theta))
    grad_array = numpy.zeros((num_features, 1))
    grad_array_reg = numpy.zeros((num_features, 1))
    for grad_index in range(0, num_features):
        grad_term = numpy.multiply(numpy.reshape(X[:, grad_index],
                                                 (num_train_ex, 1)),
                                   numpy.subtract(h_theta, y))
        grad_array[grad_index] = (numpy.sum(grad_term, axis=0))/num_train_ex
        grad_array_reg[grad_index] = (
            grad_array[grad_index]+(lamb/num_train_ex)*theta[grad_index])
    grad_array_reg[0] = grad_array_reg[0]-(lamb/num_train_ex)*theta[0]
    grad_array_reg_flat = numpy.ndarray.flatten(grad_array_reg)
    return grad_array_reg_flat


def train_log_reg(X, y):
    """ Solves for optimal logistic regression weights.
    Args:
      X: Matrix of features.
      y: Vector of labels.
    Returns:
      theta: Vector of parameters for regularized logistic regression.
    """
    num_features = X.shape[1]
    num_train_ex = X.shape[0]
    # print("num_features = %d" % num_features)
    # print("num_train_ex = %d" % num_train_ex)
    ones_vec = numpy.ones((num_train_ex, 1))
    X_aug = numpy.c_[ones_vec, X]
    y_vec = numpy.reshape(y, (num_train_ex, 1))
    theta_vec = numpy.zeros((num_features+1, 1))
    theta_vec_flat = numpy.ndarray.flatten(theta_vec)
    # f_min_ncg_out = fmin_ncg(compute_cost, theta_vec_flat,
                             # fprime=compute_gradient, args=(X_aug, y_vec,
                                                            # num_train_ex),
                             # avextol=1e-10, epsilon=1e-10, maxiter=400,
                             # full_output=1)
    lamb = 0
    # lamb = 1
    print("Running logistic regression with lamb = %.3f..." % lamb)
    f_min_ncg_out = fmin_ncg(compute_cost_reg, theta_vec_flat,
                             fprime=compute_gradient_reg, args=(X_aug, y_vec,
                                                                num_train_ex,
                                                                lamb),
                             avextol=1e-10, epsilon=1e-10, maxiter=400,
                             full_output=1)
    theta_opt = numpy.reshape(f_min_ncg_out[0], (num_features+1, 1), order='F')
    print("theta:")
    print("%s\n" % numpy.array_str(numpy.round(theta_opt, 6)))
    return theta_opt


def test_log_reg(X_test, theta_opt):
    """ Computes predictions using optimal logistic regression weights.
    Args:
      X_test: Matrix of test features.
      theta_opt: Vector of parameters for logistic regression.
    Returns:
      winning_prob: Vector of probability-based predictions.
    """
    ones_test_vec = numpy.ones((X_test.shape[0], 1))
    X_test_aug = numpy.c_[ones_test_vec, X_test]
    winning_prob = compute_sigmoid(numpy.dot(X_test_aug, theta_opt))
    return winning_prob
