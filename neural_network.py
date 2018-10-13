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

# Methods for neural network
from scipy.optimize import fmin_ncg
from compute_sigmoid import compute_sigmoid
from compute_sigmoid import comp_sig_grad
import numpy


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def rand_initialize_weights(l_in, l_out):
    """ Initializes random weights of neural network between layers $L$ and
        $L+1$.
    Args:
      l_in: Number of units in layer $L$.
      l_out: Number of units in layer $L+1$.
    Returns:
      w_mat: Matrix of random weights.
    Raises:
      An error occurs if the number of units in layer $L+1$ is 0.
    """
    if (l_out == 0): raise Error('l_out = 0')
    epsilon_init = 0.12
    w_mat = numpy.zeros((l_out, 1+l_in))
    for l_out_index in range(0, l_out):
        w_mat[l_out_index, :] = (
            2*epsilon_init*numpy.random.random((1+l_in, ))-epsilon_init)
    return w_mat


def compute_cost(theta, X, y, lamb, l_1_siz, l_2_siz, l_3_siz):
    """ Computes neural network cost.
    Args:
      theta: Vector of neural network parameters.
      X: Matrix of features.
      y: Vector of labels.
      lamb: Regularization parameter.
      l_1_siz: Number of units in input layer.
      l_2_siz: Number of units in hidden layer.
      l_3_siz: Number of units in output layer.
    Returns:
      j_theta_reg: Regularized neural network cost.
    Raises:
      An error occurs if the number of training examples is 0.
    """
    num_ex = X.shape[0]
    if (num_ex == 0): raise Error('num_ex = 0')
    theta = numpy.reshape(theta, (l_2_siz*(l_1_siz+1)+l_3_siz*(l_2_siz+1),
                                  1), order='F')
    theta_1_slice = theta[0:(l_2_siz*(l_1_siz+1)), :]
    theta_1 = numpy.reshape(theta_1_slice, (l_2_siz, l_1_siz+1), order='F')
    theta_2_slice = (
        theta[(l_2_siz*(l_1_siz+1)):(l_2_siz*(l_1_siz+1)+l_3_siz*(l_2_siz+1)),
              :])
    theta_2 = numpy.reshape(theta_2_slice, (l_3_siz, l_2_siz+1), order='F')
    ones_vec = numpy.ones((num_ex, 1))
    aug_x = numpy.c_[ones_vec, X]
    hidden_layer_activation = (
        compute_sigmoid(numpy.dot(aug_x, numpy.transpose(theta_1))))
    ones_vec_mod = numpy.ones((hidden_layer_activation.shape[0], 1))
    hidden_layer_activation_mod = numpy.c_[ones_vec_mod,
                                           hidden_layer_activation]
    output_layer_activation = (
        compute_sigmoid(numpy.dot(hidden_layer_activation_mod,
                                  numpy.transpose(theta_2))))
    num_label = theta_2.shape[0]
    y_mat = numpy.zeros((num_ex, num_label))
    for example_index in range(0, num_ex):
        col_val = y[example_index].astype(int)
        y_mat[example_index, col_val-1] = 1
    theta_1_sq = numpy.power(theta_1, 2)
    theta_2_sq = numpy.power(theta_2, 2)
    cost_term_1 = numpy.multiply(-y_mat, numpy.log(output_layer_activation))
    cost_term_2 = numpy.multiply(-(1-y_mat),
                                 numpy.log(1-output_layer_activation))
    j_theta = ((numpy.add(cost_term_1, cost_term_2)).sum())/num_ex
    j_theta_reg = (
        j_theta+(lamb/(2*num_ex))*((numpy.transpose(theta_1_sq))[1:, ].sum()+
                                   (numpy.transpose(theta_2_sq))[1:, ].sum()))
    return j_theta_reg


def compute_gradient(theta, X, y, lamb, l_1_siz, l_2_siz, l_3_siz):
    """ Computes neural network gradient via backpropagation.
    Args:
      theta: Vector of neural network parameters.
      X: Matrix of features.
      y: Vector of labels.
      lamb: Regularization parameter.
      l_1_siz: Number of units in input layer.
      l_2_siz: Number of units in hidden layer.
      l_3_siz: Number of units in output layer.
    Returns:
      grad_array_reg_flat: Vector of regularized neural network gradients (one 
                           per feature).
    Raises:
      An error occurs if the number of training examples is 0.
    """
    num_ex = X.shape[0]
    if (num_ex == 0): raise Error('num_ex = 0')
    theta = numpy.reshape(theta, (l_2_siz*(l_1_siz+1)+l_3_siz*(l_2_siz+1),
                                  1), order='F')
    theta_1_slice = theta[0:(l_2_siz*(l_1_siz+1)), :]
    theta_1 = numpy.reshape(theta_1_slice, (l_2_siz, l_1_siz+1), order='F')
    theta_2_slice = (
        theta[(l_2_siz*(l_1_siz+1)):(l_2_siz*(l_1_siz+1)+l_3_siz*(l_2_siz+1)),
              :])
    theta_2 = numpy.reshape(theta_2_slice, (l_3_siz, l_2_siz+1), order='F')
    ones_vec = numpy.ones((num_ex, 1))
    aug_x = numpy.c_[ones_vec, X]
    delta_1_mat = numpy.zeros((theta_1.shape[0], aug_x.shape[1]))
    delta_2_mat = numpy.zeros((theta_2.shape[0], theta_1.shape[0]+1))
    num_label = theta_2.shape[0]

    # Iterate over the training examples
    for example_index in range(0, num_ex):

        # Step 1
        example_x = aug_x[example_index:(example_index+1), :]
        hidden_layer_activation = (
            compute_sigmoid(numpy.dot(example_x, numpy.transpose(theta_1))))
        ones_vec_mod = numpy.ones((hidden_layer_activation.shape[0], 1))
        hidden_layer_activation_mod = numpy.c_[ones_vec_mod,
                                               hidden_layer_activation]
        output_layer_activation = (
            compute_sigmoid(numpy.dot(hidden_layer_activation_mod,
                                      numpy.transpose(theta_2))))

        # Step 2
        y_vec = numpy.zeros((1, num_label))
        col_val = y[example_index].astype(int)
        y_vec[:, col_val-1] = 1
        delta_3_vec = numpy.transpose(numpy.subtract(output_layer_activation,
                                                     y_vec))

        # Step 3
        delta_2_int = numpy.dot(numpy.transpose(theta_2), delta_3_vec)
        t_1_tran = numpy.transpose(theta_1)
        delta_2_vec = (
            numpy.multiply(delta_2_int[1:, ],
                           comp_sig_grad(numpy.transpose(numpy.dot(example_x,
                                                                   t_1_tran)))))

        # Step 4
        delta_1_mat = numpy.add(delta_1_mat, numpy.dot(delta_2_vec, example_x))
        delta_2_mat = numpy.add(delta_2_mat,
                                numpy.dot(delta_3_vec,
                                          numpy.c_[1, hidden_layer_activation]))

    # Step 5 (without regularization)
    theta_1_grad = (1/num_ex)*delta_1_mat
    theta_2_grad = (1/num_ex)*delta_2_mat

    # Step 5 (with regularization)
    theta_1_grad[:, 1:] = theta_1_grad[:, 1:]+(lamb/num_ex)*theta_1[:, 1:]
    theta_2_grad[:, 1:] = theta_2_grad[:, 1:]+(lamb/num_ex)*theta_2[:, 1:]

    # Unroll gradients
    theta_1_grad_stack = theta_1_grad.flatten(1)
    theta_2_grad_stack = theta_2_grad.flatten(1)
    grad_array_reg = numpy.concatenate((theta_1_grad_stack,
                                        theta_2_grad_stack), axis=None)
    grad_array_reg_flat = numpy.ndarray.flatten(grad_array_reg)
    return grad_array_reg_flat


def predict(theta_1, theta_2, X):
    """ Performs label prediction on training data.
    Args:
      theta_1: Matrix of neural network parameters (map from input layer to
               hidden layer).
      theta_2: Matrix of neural network parameters (map from hidden layer to
               output layer).
      X: Matrix of features.
    Returns:
      p: Vector of probabilities (one per example).
    Raises:
      An error occurs if the number of training examples is 0.
    """
    num_train_ex = X.shape[0]
    if (num_train_ex == 0): raise Error('num_train_ex = 0')
    ones_vec = numpy.ones((num_train_ex, 1))
    aug_x = numpy.c_[ones_vec, X]
    hidden_layer_activation = (
        compute_sigmoid(numpy.dot(aug_x, numpy.transpose(theta_1))))
    ones_vec_mod = numpy.ones((hidden_layer_activation.shape[0], 1))
    hidden_layer_activation_mod = numpy.c_[ones_vec_mod,
                                           hidden_layer_activation]
    output_layer_activation = (
        compute_sigmoid(numpy.dot(hidden_layer_activation_mod,
                                  numpy.transpose(theta_2))))
    return output_layer_activation[:, 0]


def train_and_test_nn(X_train, y_train, X_test, X_curr):
    """ Trains and tests neural network.
    Args:
      X_train: Matrix of training features.
      y_train: Vector of training labels.
      X_test: Matrix of testing features.
      X_curr: Additional matrix of testing features.
    Returns:
      return_list: List of two objects.
                   test_prob: Vector of predictions for test data.
                   curr_prob: Additional vector of predictions for test data.
    """
    print("Training neural network ...")
    i_l_siz = X_train.shape[1]
    h_l_siz = 26
    num_label = 2
    init_theta_1 = rand_initialize_weights(i_l_siz, h_l_siz)
    init_theta_2 = rand_initialize_weights(h_l_siz, num_label)
    init_theta_1_stack = init_theta_1.flatten(1)
    init_theta_2_stack = init_theta_2.flatten(1)
    theta_stack = numpy.concatenate((init_theta_1_stack, init_theta_2_stack),
                                    axis=None)
    theta_stack_flat = numpy.ndarray.flatten(theta_stack)
    lamb = 1
    f_min_ncg_out = fmin_ncg(compute_cost, theta_stack_flat,
                             fprime=compute_gradient,
                             args=(X_train, y_train, lamb, i_l_siz, h_l_siz,
                                   num_label), avextol=1e-10,
                             epsilon=1e-10, maxiter=20, full_output=1)
    t_opt = numpy.reshape(f_min_ncg_out[0],
                          (h_l_siz*(i_l_siz+1)+num_label*(h_l_siz+1), 1),
                          order='F')
    theta_1_slice = t_opt[0:(h_l_siz*(i_l_siz+1)), :]
    theta_1_mat = numpy.reshape(theta_1_slice, (h_l_siz, i_l_siz+1), order='F')
    theta_2_slice = (
        t_opt[(h_l_siz*(i_l_siz+1)):(h_l_siz*(i_l_siz+1)+num_label*(h_l_siz+1)),
              :])
    theta_2_mat = numpy.reshape(theta_2_slice, (num_label, h_l_siz+1),
                                order='F')
    test_prob = predict(theta_1_mat, theta_2_mat, X_test)
    curr_prob = predict(theta_1_mat, theta_2_mat, X_curr)
    return_list = {'test_prob': test_prob,
                   'curr_prob': curr_prob}
    return return_list
