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

# Method for computing sigmoid function
import numpy


def compute_sigmoid(z):
    """ Computes sigmoid function.
    Args:
      z: Can be a scalar, a vector or a matrix.
    Returns:
      sigmoid_z: Sigmoid function value.
    """
    sigmoid_z = 1/(1+numpy.exp(-z))
    return sigmoid_z


def comp_sig_grad(z):
    """ Computes gradient of sigmoid function.
    Args:
      z: Can be a scalar, a vector or a matrix.
    Returns:
      sigmoid_gradient_z: Sigmoid gradient function value.
    """
    sigmoid_gradient_z = numpy.multiply(compute_sigmoid(z),
                                        1-compute_sigmoid(z))
    return sigmoid_gradient_z
