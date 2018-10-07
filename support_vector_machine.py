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

# Methods for support vector machine
from sklearn import svm
import numpy


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def train_and_test_svm(X_train, y_train, X_test, X_curr):
    """ Solves for SVM boundary.
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
    # print("Training Linear SVM ...")
    # svm_model = svm.SVC(kernel='linear',probability=True)
    print("Training SVM with RBF Kernel (this may take 1 to 2 minutes) ...")
    # sigma_val = 0.1
    sigma_val = 5
    svm_model = svm.SVC(kernel='rbf', gamma=1/(2*numpy.power(sigma_val, 2)),
                        probability=True)
    svm_model.fit(X_train, y_train)
    test_prob = svm_model.predict_proba(X_test)
    curr_prob = svm_model.predict_proba(X_curr)
    return_list = {'test_prob': test_prob[:, 1],
                   'curr_prob': curr_prob[:, 1]}
    return return_list
