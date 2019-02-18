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

# Ensemble method
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import numpy


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def train_and_test_em(X_train, y_train, X_test, X_curr):
    """ Uses ensemble method to learn decision boundaries.
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
    # print("Training ensemble method ...")
    # em = AdaBoostClassifier(random_state=None)
    em = BaggingClassifier(LogisticRegression(random_state=None),
                           random_state=None)
    # em = BaggingClassifier(SVC(kernel='linear',probability=True),
    #                        random_state=None)
    # em = BaggingClassifier(SVC(kernel='poly', probability=True),
    #                        random_state=None)
    # em = BaggingClassifier(SVC(kernel='rbf', probability=True),
    #                        random_state=None)
    # em = BaggingClassifier(SVC(kernel='sigmoid', probability=True),
    #                        random_state=None)
    # em = ExtraTreesClassifier(random_state=None)
    # em = GradientBoostingClassifier(random_state=None)
    # em = RandomForestClassifier(random_state=None)
    ensemble_method = em.fit(X_train, y_train)
    test_prob = ensemble_method.predict_proba(X_test)
    curr_prob = ensemble_method.predict_proba(X_curr)
    return_test_prob = numpy.reshape(test_prob[:, 1], (X_test.shape[0], 1))
    return_curr_prob = numpy.reshape(curr_prob[:, 1], (X_curr.shape[0], 1))
    return_list = {'test_prob': return_test_prob,
                   'curr_prob': return_curr_prob}
    return return_list
