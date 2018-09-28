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

# Machine Learning March Madness
# Apply ML methods to predict outcome of 2014 NCAA Tournament
from matplotlib import pyplot
from scipy.optimize import fmin_ncg
from gen_elo_differential import gen_elo_differential
from gen_point_differential import gen_point_differential
from gen_seed_difference import gen_seed_difference
import numpy
import itertools
import math


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def feature_normalize(x):
    """ Performs feature normalization.
    Args:
      x: Vector of training data.
    Returns:
      y: Vector of normalized training data.
    Raises:
      An error occurs if the number of training examples is 0.
    """
    num_ex = x.shape[0]
    if num_ex == 0: raise Error('num_ex = 0')
    x_mean = numpy.mean(x)
    x_std = numpy.std(x)
    x_scale = (x-x_mean)/x_std
    return x_scale


def plot_data(x, y):
    """ Plots data.
    Args:
      x: Features to be plotted.
      y: Data labels.
    Returns:
      None.
    """
    positive_indices = numpy.where(y == 1)
    negative_indices = numpy.where(y == 0)
    pos = pyplot.scatter(x[positive_indices, 0], x[positive_indices, 1], s=80,
                         marker='+', color='k')
    pyplot.hold(True)
    neg = pyplot.scatter(x[negative_indices, 0], x[negative_indices, 1], s=80,
                         marker='s', color='y')
    pyplot.legend((pos, neg), ('y = 1', 'y = 0'), loc='lower right')
    pyplot.hold(False)
    pyplot.ylabel('Seed Difference', fontsize=18)
    pyplot.xlabel('Net Point Differential', fontsize=18)
    return None


def plot_decision_boundary(x, y, theta):
    """ Plots decision boundary.
    Args:
      x: Features that have already been plotted.
      y: Data labels.
      theta: Parameter that determines slope of decision boundary.
    Returns:
      None.
    """
    plot_data(x, y)
    pyplot.hold(True)
    y_line_vals = (theta[0]+theta[1]*x[:, 0])/(-1*theta[2])
    pyplot.plot(x[:, 0], y_line_vals, 'b-', markersize=18)
    pyplot.hold(False)
    return None


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


def gen_train_results(prev_tourney_results):
    """ Generates matrix of training results by parsing data from previous
        tournaments.
    Args:
      prev_tourney_results: Matrix of previous tournament results that consists
                            of these columns:
                            Column 1: character denoting season ID
                            Column 2: integer denoting ID of date of game
                            Column 3: integer denoting ID of winning team
                            Column 4: integer denoting score of winning team
                            Column 5: integer denoting ID of losing team
                            Column 6: integer denoting score of losing team
                            Column 7: number of overtime periods
    Returns:
      training_data: Matrix that consists of these columns:
                     Column 1: integer denoting season ID
                     Column 2: integer denoting ID of team A
                     Column 3: integer denoting ID of team B (assume that in
                     each row, value in Column 3 exceeds value in Column 2)
                     Column 4: 0 if team A lost to team B; otherwise, 1 (assume
                     that A and B played in that season's tournament)
                     Column(s) of training features will be added by other
                     functions
    """
    num_prev_tourney_games = prev_tourney_results.shape[0]-1
    training_data = numpy.zeros((num_prev_tourney_games, 4))
    for prev_tourney_game_idx in range(0, num_prev_tourney_games):
        season_id = prev_tourney_results[prev_tourney_game_idx+1, 0]
        winning_team_id = prev_tourney_results[prev_tourney_game_idx+1, 2]
        losing_team_id = prev_tourney_results[prev_tourney_game_idx+1, 4]
        if (winning_team_id < losing_team_id):
            team_A = winning_team_id
            team_B = losing_team_id
            outcome = 1
        else:
            team_A = losing_team_id
            team_B = winning_team_id
            outcome = 0
        training_data[prev_tourney_game_idx, 0] = ord(season_id)
        training_data[prev_tourney_game_idx, 1] = team_A
        training_data[prev_tourney_game_idx, 2] = team_B
        training_data[prev_tourney_game_idx, 3] = outcome
    return training_data


def coin_flip(team_ids):
    """ Generates coin-flip predictions.
    Args:
      team_ids: Vector of team IDs.
    Returns:
      pred_mat: Matrix of coin-flip predictions, where first and second columns
                include IDs of distinct teams; each entry in third column is
                0.5.  Only unordered pairings of teams appear in this matrix.
    """
    team_ids_list = team_ids.tolist()
    team_id_pairs = itertools.combinations(team_ids_list, 2)
    team_id_pairs_array = numpy.asarray(list(team_id_pairs))
    coin_flips = 0.5*numpy.ones((team_id_pairs_array.shape[0], 1))
    pred_mat = numpy.concatenate((team_id_pairs_array, coin_flips), axis=1)
    return pred_mat


def gen_raw_submission(file_name, pred_mat):
    """ Generates raw submission file.
    Args:
      file_name: Name of raw submission file.
      pred_mat: Matrix of predictions, where first and second columns include
                IDs of distinct teams; each entry in third column is probability
                that team in first column defeats team in second column.  Only
                unordered pairings of teams appear in this matrix.
    Returns:
      None.
    """
    num_pairs = pred_mat.shape[0]
    file_mat = numpy.zeros((num_pairs+1, 2), dtype=object)
    file_mat[0, 0] = "id"
    file_mat[0, 1] = "pred"
    for pair_idx in range(0, num_pairs):
        id1 = pred_mat[pair_idx, 0].astype(float)
        id2 = pred_mat[pair_idx, 1].astype(float)
        curr_id = "S_%d_%d" % (id1, id2)
        file_mat[pair_idx+1, 0] = curr_id
        file_mat[pair_idx+1, 1] = str(pred_mat[pair_idx, 2])
    numpy.savetxt(file_name, file_mat, fmt='%s', delimiter=',')


def parse_raw_submission(raw_submission):
    """ Parses raw submission input and returns matrix.
    Args:
      raw_submission: Matrix of raw submission input, where first column
                      includes entries of the form "S_A_B" where A and B are the
                      IDs of two teams; second column includes probabilities
                      that team A will defeat team B.
    Returns:
      parsed_submission: Matrix with three columns, where first and second
                         columns include IDs of above-mentioned teams A and B,
                         respectively; third column includes above-mentioned
                         probabilities.
    """
    raw_games = raw_submission[:, 0]
    num_games = raw_games.shape[0]-1
    id1 = numpy.zeros((num_games, 1))
    id2 = numpy.zeros((num_games, 1))
    for game_index in range(1, num_games+1):
        curr_game = raw_games[game_index]
        curr_game_string = numpy.array2string(curr_game)
        S,tmp_id1,tmp_id2 = curr_game_string.split("_")
        id1[game_index-1] = int(tmp_id1)
        tmp2_id2,tmp3_id2 = tmp_id2.split("'")
        id2[game_index-1] = tmp2_id2
    submit_probs = raw_submission[1:(num_games+1), 1]
    parsed_probs = submit_probs.reshape((num_games, 1))
    parsed_submission = numpy.concatenate((id1, id2, parsed_probs), axis=1)
    return parsed_submission


def evaluate_submission(results, submission, season_ids):
    """ Computes predicted binomial deviance between results and submission.
    Args:
      results: Matrix of tournament results, where each row represents a game;
               third and fifth columns include IDs of winning and losing teams,
               respectively.
      submission: Matrix of submissions, where each row represents a game; first
                  two columns include IDs of two teams and third column includes
                  probability that first team wins.
      season_ids: Array of integers denoting IDs of seasons for submission
    Returns:
      log_loss: Predicted binomial deviance.
    """
    bound_extreme = 0.0000000000000020278
    str_seasons = results[1:, 0]
    seasons = numpy.zeros((str_seasons.shape[0], 1))
    for season_index in range(0, str_seasons.shape[0]):
        seasons[season_index] = ord(str_seasons[season_index])
    winners = results[1:, 2].astype(int)
    losers = results[1:, 4].astype(int)
    submission_team1 = submission[:, 0].astype(float)
    submission_team2 = submission[:, 1].astype(float)
    submission_prob = submission[:, 2].astype(float)
    num_games = results.shape[0]-1
    log_loss_array = numpy.zeros((num_games, 1))
    num_seasons = len(season_ids)
    num_submissions = submission.shape[0]
    num_subs_per_season = num_submissions/num_seasons
    curr_game_index = 0
    for season_index in range(0, num_seasons):
        curr_season = season_ids[season_index]
        start_index = season_index*num_subs_per_season
        end_index = (season_index+1)*num_subs_per_season
        curr_sub_team1 = submission_team1[start_index:end_index]
        curr_sub_team2 = submission_team2[start_index:end_index]
        curr_sub_prob = submission_prob[start_index:end_index]
        season_game_index = numpy.where(seasons == curr_season)
        curr_winners = winners[season_game_index[0]]
        curr_losers = losers[season_game_index[0]]
        curr_num_games = curr_winners.shape[0]
        for game_index in range(0, curr_num_games):
            curr_winner = curr_winners[game_index]
            curr_loser = curr_losers[game_index]

            # Find game involving this winner and loser in submission
            if (curr_winner < curr_loser):
                curr_outcome = 1
                winner_index = numpy.where(curr_sub_team1 == curr_winner)
                loser_index = numpy.where(curr_sub_team2 == curr_loser)
            else:
                curr_outcome = 0
                loser_index = numpy.where(curr_sub_team1 == curr_loser)
                winner_index = numpy.where(curr_sub_team2 == curr_winner)
            submission_index = numpy.intersect1d(winner_index[0],
                                                 loser_index[0])
            curr_prob = curr_sub_prob[submission_index].astype(float)

            # Bound prediction from extremes if necessary
            if (curr_prob == 1):
                curr_prob = 1-bound_extreme
            elif (curr_prob == 0):
                curr_prob = bound_extreme

            # Evaluate per-game log loss
            log_loss_term1 = curr_outcome*math.log(curr_prob)
            log_loss_term2 = (1-curr_outcome)*math.log(1-curr_prob)
            log_loss_array[curr_game_index] = log_loss_term1+log_loss_term2
            curr_game_index = curr_game_index+1
        curr_log_loss = (-1/curr_game_index)*numpy.sum(log_loss_array)
        # print("curr_log_loss = %f" % curr_log_loss)

    # Compute log loss
    log_loss = (-1/num_games)*numpy.sum(log_loss_array)
    return log_loss


def main():
    """ Main function
    """
    print("Loading list of teams.")
    teams = numpy.genfromtxt("teams.csv", delimiter=",")
    team_ids = teams[1:, 0]
    print("Loading regular season results.")
    regular_season_results = numpy.genfromtxt("regular_season_results.csv",
                                              dtype=object, delimiter=",")
    print("Loading train tournament results.")
    train_tourney_results = numpy.genfromtxt("tourney_results_train.csv",
                                             dtype=object, delimiter=",")
    print("Loading test tournament results.")
    test_tourney_results = numpy.genfromtxt("tourney_results_test.csv",
                                            dtype=object, delimiter=",")
    print("Loading current tournament results.")
    curr_tourney_results = numpy.genfromtxt("tourney_results_2014.csv",
                                            dtype=object, delimiter=",")
    print("Loading tournament seeds.")
    tournament_seeds = numpy.genfromtxt("tourney_seeds.csv", dtype=str,
                                        delimiter=",")

    # Generate training results
    training_mat = gen_train_results(train_tourney_results)

    # Logistic regression algorithm
    curr_const = 0.001
    test_season_ids = [67, 72, 77, 82]
    num_test_seasons = len(test_season_ids)
    curr_season_id = 83

    # Compute Elo differential between teams A and B for each season
    elo_diff_mat_list = gen_elo_differential(regular_season_results, team_ids,
                                             training_mat, test_season_ids,
                                             curr_season_id)
    elo_diff_mat = elo_diff_mat_list['elo_diff_mat']
    elo_idx1 = numpy.where(elo_diff_mat[:, 4] != curr_const)
    elo_idx = elo_idx1[0]

    # Compute point differential between teams A and B for each season
    point_diff_mat_list = gen_point_differential(regular_season_results,
                                                 team_ids, training_mat,
                                                 test_season_ids,
                                                 curr_season_id, curr_const)
    point_diff_mat = point_diff_mat_list['point_diff_mat']
    point_idx1 = numpy.where(point_diff_mat[:, 4] != curr_const)
    point_idx = point_idx1[0]

    # Compute seed difference between teams A and B for each season (where teams
    # A and B are both in that season's tournament)
    seed_diff_mat_list = gen_seed_difference(tournament_seeds, team_ids,
                                             training_mat, test_season_ids,
                                             curr_season_id, curr_const)
    seed_diff_mat = seed_diff_mat_list['seed_diff_mat']
    seed_idx1 = numpy.where(seed_diff_mat[:, 4] != curr_const)
    seed_idx = seed_idx1[0]

    # Set training features and labels
    # focus_idx = elo_idx
    focus_idx = point_idx
    # focus_idx = seed_idx
    feature_elo_scale = feature_normalize(elo_diff_mat[focus_idx, 4])
    feature_point_scale = feature_normalize(point_diff_mat[focus_idx, 4])
    feature_seed_scale = feature_normalize(seed_diff_mat[focus_idx, 4])
    x_mat = numpy.c_[feature_elo_scale, feature_point_scale]
    # x_mat = numpy.c_[feature_elo_scale, feature_seed_scale]
    # x_mat = numpy.c_[feature_point_scale, feature_seed_scale]
    # x_mat = numpy.c_[feature_elo_scale, feature_point_scale, feature_seed_scale]
    # x_mat = numpy.reshape(feature_elo_scale, (feature_elo_scale.shape[0], 1))
    # x_mat = numpy.reshape(feature_point_scale,
    #                       (feature_point_scale.shape[0], 1))
    # x_mat = numpy.reshape(feature_seed_scale,
    #                       (feature_seed_scale.shape[0], 1))
    label_vec = elo_diff_mat[focus_idx, 3]

    # Run nonconjugate gradient algorithm
    num_features = x_mat.shape[1]
    num_train_ex = x_mat.shape[0]
    # print("num_features = %d" % num_features)
    # print("num_train_ex = %d" % num_train_ex)
    ones_vec = numpy.ones((num_train_ex, 1))
    x_mat_aug = numpy.c_[ones_vec, x_mat]
    y_vec = numpy.reshape(label_vec, (num_train_ex, 1))
    theta_vec = numpy.zeros((num_features+1, 1))
    theta_vec_flat = numpy.ndarray.flatten(theta_vec)
    # f_min_ncg_out = fmin_ncg(compute_cost, theta_vec_flat,
                             # fprime=compute_gradient, args=(x_mat_aug, y_vec,
                                                            # num_train_ex),
                             # avextol=1e-10, epsilon=1e-10, maxiter=400,
                             # full_output=1)
    # lamb = 0.3
    lamb = 1
    f_min_ncg_out = fmin_ncg(compute_cost_reg, theta_vec_flat,
                             fprime=compute_gradient_reg, args=(x_mat_aug,
                                                                y_vec,
                                                                num_train_ex,
                                                                lamb),
                             avextol=1e-10, epsilon=1e-10, maxiter=400,
                             full_output=1)
    theta_opt = numpy.reshape(f_min_ncg_out[0], (num_features+1, 1), order='F')
    print("theta:")
    print("%s\n" % numpy.array_str(numpy.round(theta_opt, 6)))

    # Plot point differential and seed difference between teams A and B
    # return_code = plot_decision_boundary(x_mat, label_vec, theta_opt)
    # pyplot.legend(('Decision Boundary', 'Team B Wins', 'Team A Wins'),
    #               loc='lower right')
    # pyplot.show()

    # Compute predictions on test data
    elo_diff_test_season = elo_diff_mat_list['test_season_mat']
    point_diff_test_season = point_diff_mat_list['test_season_mat']
    seed_diff_test_season = seed_diff_mat_list['test_season_mat']
    test_feature_elo_scale = feature_normalize(elo_diff_test_season[:, 3])
    test_feature_point_scale = feature_normalize(point_diff_test_season[:, 3])
    test_feature_seed_scale = feature_normalize(seed_diff_test_season[:, 3])
    test_mat = numpy.c_[test_feature_elo_scale, test_feature_point_scale]
    # test_mat = numpy.c_[test_feature_elo_scale, test_feature_seed_scale]
    # test_mat = numpy.c_[test_feature_point_scale, test_feature_seed_scale]
    # test_mat = numpy.c_[test_feature_elo_scale, test_feature_point_scale,
    #                     test_feature_seed_scale]
    # test_mat = numpy.reshape(test_feature_elo_scale,
    #                          (test_feature_elo_scale.shape[0], 1))
    # test_mat = numpy.reshape(test_feature_point_scale,
    #                          (test_feature_point_scale.shape[0], 1))
    # test_mat = numpy.reshape(test_feature_seed_scale,
    #                          (test_feature_seed_scale.shape[0], 1))
    ones_test_vec = numpy.ones((test_mat.shape[0], 1))
    x_test_mat = test_mat
    x_test_mat_aug = numpy.c_[ones_test_vec, x_test_mat]
    winning_prob = compute_sigmoid(numpy.dot(x_test_mat_aug, theta_opt))
    init_pred_mat = elo_diff_test_season[:, 1:4]
    test_pred_mat = numpy.c_[init_pred_mat[:, 0:2], winning_prob]

    # Coin-flip algorithm
    # pred_mat = coin_flip(team_ids)

    # Generate raw submission file
    test_file_name = "test_submission.csv"
    gen_raw_submission(test_file_name, test_pred_mat)

    # Load raw submission file
    print("Loading test submission file.")
    test_raw_submission = numpy.genfromtxt(test_file_name, dtype=None,
                                           delimiter=",")

    # Parse raw submission file
    test_submission = parse_raw_submission(test_raw_submission)

    # Compute evaluation function of submission
    test_log_loss = evaluate_submission(test_tourney_results, test_submission,
                                        test_season_ids)
    print("Test binomial deviance:  %.5f" % test_log_loss)

    # Compute predictions on current data
    elo_diff_curr_season = elo_diff_mat_list['curr_season_mat']
    point_diff_curr_season = point_diff_mat_list['curr_season_mat']
    seed_diff_curr_season = seed_diff_mat_list['curr_season_mat']
    curr_feature_elo_scale = feature_normalize(elo_diff_curr_season[:, 3])
    curr_feature_point_scale = feature_normalize(point_diff_curr_season[:, 3])
    curr_feature_seed_scale = feature_normalize(seed_diff_curr_season[:, 3])
    curr_mat = numpy.c_[curr_feature_elo_scale, curr_feature_point_scale]
    # curr_mat = numpy.c_[curr_feature_elo_scale, curr_feature_seed_scale]
    # curr_mat = numpy.c_[curr_feature_point_scale, curr_feature_seed_scale]
    # curr_mat = numpy.c_[curr_feature_elo_scale, curr_feature_point_scale,
    #                     curr_feature_seed_scale]
    # curr_mat = numpy.reshape(curr_feature_elo_scale,
    #                          (curr_feature_elo_scale.shape[0], 1))
    # curr_mat = numpy.reshape(curr_feature_point_scale,
    #                          (curr_feature_point_scale.shape[0], 1))
    # curr_mat = numpy.reshape(curr_feature_seed_scale,
    #                          (curr_feature_seed_scale.shape[0], 1))
    ones_curr_vec = numpy.ones((curr_mat.shape[0], 1))
    x_curr_mat = curr_mat
    x_curr_mat_aug = numpy.c_[ones_curr_vec, x_curr_mat]
    winning_prob = compute_sigmoid(numpy.dot(x_curr_mat_aug, theta_opt))
    init_pred_mat = coin_flip(team_ids)
    curr_pred_mat = numpy.c_[init_pred_mat[:, 0:2], winning_prob]

    # Generate raw submission file
    curr_file_name = "curr_submission.csv"
    gen_raw_submission(curr_file_name, curr_pred_mat)

    # Load raw submission file
    print("Loading curr submission file.")
    curr_raw_submission = numpy.genfromtxt(curr_file_name, dtype=None,
                                           delimiter=",")

    # Parse raw submission file
    curr_submission = parse_raw_submission(curr_raw_submission)

    # Compute evaluation function of submission
    curr_log_loss = evaluate_submission(curr_tourney_results, curr_submission,
                                        [curr_season_id])
    print("Current binomial deviance:  %.5f" % curr_log_loss)

# Call main function
if __name__ == "__main__":
    main()
