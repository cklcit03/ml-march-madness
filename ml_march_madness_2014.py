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
from scipy.optimize import fmin_ncg
import numpy
import itertools
import math


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


def gen_point_differential(regular_season_results, team_ids, training_data,
                           current_season_id):
    """ Generates matrix of point differentials between teams A and B for each
        season of interest.

    Args:
      regular_season_results: Matrix of regular season results that
                              consists of these columns:
                              Column 1: character denoting season ID
                              Column 2: integer denoting ID of date of game
                              Column 3: integer denoting ID of winning team
                              Column 4: integer denoting score of winning team
                              Column 5: integer denoting ID of losing team
                              Column 6: integer denoting score of losing team
                              Column 7: character denoting location of winning
                                        team
                              Column 8: number of overtime periods
      team_ids: Vector of team IDs.
      training_data: Matrix that consists of these columns:
                     Column 1: character denoting season ID
                     Column 2: integer denoting ID of team A
                     Column 3: integer denoting ID of team B (assume that in
                     each row, value in Column 3 exceeds value in Column 2)
                     Column 4: 0 if team A lost to team B; otherwise, 1 (assume
                     that A and B played in that season's tournament)
      current_season_id: Integer denoting ID of current season

    Returns:
      return_list: List of two objects.
                   point_diff_mat: Matrix that consists of these columns:
                                   Column 1: integer denoting season ID
                                   Column 2: integer denoting ID of team A
                                   Column 3: integer denoting ID of team B
                                   (assume that in each row, value in Column 3
                                   exceeds value in Column 2)
                                   Column 4: 0 if team A lost to team B;
                                   otherwise, 1 (assume that A and B played in
                                   that season's tournament)
                                   Column 5: difference between point
                                   differential of team A and point differential
                                   of team B for this season
                   curr_season_mat: Matrix that consists of these columns:
                                    Column 1: integer denoting current season ID
                                    Column 2: integer denoting ID of team A
                                    Column 3: integer denoting ID of team B
                                    (assume that in each row, value in Column 3
                                    exceeds value in Column 2)
                                    Column 4: difference between point
                                    differential of team A and point
                                    differential of team B for current season
    """
    curr_const = 0.001
    regular_season_results_no_header = regular_season_results[1:, :]
    season_ids = regular_season_results_no_header[:, 0]
    unique_season_ids = numpy.unique(season_ids)
    num_unique_seasons = unique_season_ids.shape[0]
    tmp_vec = curr_const*numpy.ones((training_data.shape[0], 1))
    point_diff_mat = numpy.c_[training_data, tmp_vec]
    for season_idx in range(0, num_unique_seasons):
        game_indices = numpy.where(season_ids == unique_season_ids[season_idx])
        season_results = regular_season_results_no_header[game_indices[0], :]

        # For each season, compute point differential for each team
        winner_ids = season_results[:, 2].astype(float)
        winner_scores = season_results[:, 3].astype(float)
        loser_ids = season_results[:, 4].astype(float)
        loser_scores = season_results[:, 5].astype(float)
        net_differential = curr_const*numpy.ones((team_ids.shape[0], 1))
        for team_idx in range(0, team_ids.shape[0]):
            curr_team = team_ids[team_idx].astype(float)
            win_indices = numpy.where(winner_ids == curr_team)
            loss_indices = numpy.where(loser_ids == curr_team)
            if (len(win_indices[0]) > 0) and (len(loss_indices[0]) > 0):
                win_diff = numpy.subtract(winner_scores[win_indices[0]],
                                          loser_scores[win_indices[0]])
                loss_diff = numpy.subtract(winner_scores[loss_indices[0]],
                                           loser_scores[loss_indices[0]])
                total_win_diff = numpy.sum(win_diff)
                total_loss_diff = numpy.sum(loss_diff)
                # total_win_diff = numpy.mean(win_diff)
                # total_loss_diff = numpy.mean(loss_diff)
                net_differential[team_idx] = total_win_diff-total_loss_diff

        # For each season, consider all (team A, team B) pairings where teams A
        # and B played each other in the tournament
        # Compute difference between point differentials of teams A and B
        season_id = ord(unique_season_ids[season_idx])
        if (season_id != current_season_id): 
            season_idx = numpy.where((point_diff_mat[:, 0] == season_id))
            for pair_idx in season_idx[0]:
                idA = point_diff_mat[pair_idx, 1]
                idB = point_diff_mat[pair_idx, 2]
                idA_idx = numpy.where(team_ids == idA)
                idB_idx = numpy.where(team_ids == idB)
                net_diffA = net_differential[idA_idx[0]]
                net_diffB = net_differential[idB_idx[0]]
                if (net_diffA != curr_const) and (net_diffB != curr_const):
                    point_diff_mat[pair_idx, 4] = net_diffA-net_diffB
        else:
            team_ids_list = team_ids.tolist()
            team_id_pairs = itertools.combinations(team_ids_list, 2)
            team_id_pairs_array = numpy.asarray(list(team_id_pairs))
            curr_season_mat = numpy.zeros((team_id_pairs_array.shape[0], 4))
            for pair_idx in range(0, team_id_pairs_array.shape[0]):
                idA = team_id_pairs_array[pair_idx, 0]
                idB = team_id_pairs_array[pair_idx, 1]
                idA_idx = numpy.where(team_ids == idA)
                idB_idx = numpy.where(team_ids == idB)
                net_diffA = net_differential[idA_idx[0]]
                net_diffB = net_differential[idB_idx[0]]
                curr_season_mat[pair_idx, 0] = current_season_id
                curr_season_mat[pair_idx, 1] = idA
                curr_season_mat[pair_idx, 2] = idB
                curr_season_mat[pair_idx, 3] = net_diffA-net_diffB
    return_list = {'point_diff_mat': point_diff_mat,
                   'curr_season_mat': curr_season_mat}
    return return_list


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


def evaluate_submission(results, submission):
    """ Computes predicted binomial deviance between results and submission.

    Args:
      results: Matrix of tournament results, where each row represents a game;
               third and fifth columns include IDs of winning and losing teams,
               respectively.
      submission: Matrix of submissions, where each row represents a game; first
                  column includes IDs of two teams and second column includes
                  probability that first team wins.

    Returns:
      log_loss: Predicted binomial deviance.
    """
    bound_extreme = 0.0000000000000020278
    winners = results[:, 2]
    losers = results[:, 4]
    submission_team1 = submission[:, 0].flatten()
    submission_team2 = submission[:, 1].flatten()
    num_games = results.shape[0]-1
    log_loss_array = numpy.zeros((num_games, 1))
    for game_index in range(1, (num_games+1)):
        curr_winner = winners[game_index]
        curr_loser = losers[game_index]

        # Find game involving this winner and loser in submission
        if (curr_winner < curr_loser):
            curr_outcome = 1
            winner_index = numpy.where(submission_team1.astype(float) ==
                                       curr_winner)
            loser_index = numpy.where(submission_team2.astype(float) ==
                                      curr_loser)
        else:
            curr_outcome = 0
            loser_index = numpy.where(submission_team1.astype(float) ==
                                      curr_loser)
            winner_index = numpy.where(submission_team2.astype(float) ==
                                       curr_winner)
        submission_index = numpy.intersect1d(winner_index[0], loser_index[0])
        curr_prob = submission[submission_index, 2].astype(float)

        # Bound prediction from extremes if necessary
        if (curr_prob == 1):
            curr_prob = 1-bound_extreme
        elif (curr_prob == 0):
            curr_prob = bound_extreme

        # Evaluate per-game log loss
        log_loss_term1 = curr_outcome*math.log(curr_prob)
        log_loss_term2 = (1-curr_outcome)*math.log(1-curr_prob)
        log_loss_array[game_index-1] = log_loss_term1+log_loss_term2

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
    print("Loading previous tournament results.")
    prev_tourney_results = numpy.genfromtxt("tourney_results.csv",
                                            dtype=object, delimiter=",")

    # Generate training results
    training_mat = gen_train_results(prev_tourney_results)

    # Logistic regression algorithm
    # Compute point differential between teams A and B for each season (except
    # for current season)
    file_name = "point_differential.csv"
    current_season_id = 83
    point_diff_mat_list = gen_point_differential(regular_season_results,
                                                 team_ids, training_mat,
                                                 current_season_id)
    point_diff_mat = point_diff_mat_list['point_diff_mat']
    train_idx1 = numpy.where(point_diff_mat[:, 0] != current_season_id)
    train_idx2 = numpy.where(point_diff_mat[:, 4] != 0.001)
    training_indices = numpy.intersect1d(train_idx1[0], train_idx2[0])
    training_mat_aug = point_diff_mat[training_indices, :]
    num_features = 1
    num_train_ex = training_mat_aug.shape[0]
    ones_vec = numpy.ones((training_mat_aug.shape[0], 1))
    x_mat = training_mat_aug[:, 4]
    x_mat_aug = numpy.c_[ones_vec, x_mat]
    y_vec = numpy.reshape(training_mat_aug[:, 3], (num_train_ex, 1))
    theta_vec = numpy.zeros((num_features+1, 1))
    theta_vec_flat = numpy.ndarray.flatten(theta_vec)
    f_min_ncg_out = fmin_ncg(compute_cost, theta_vec_flat,
                             fprime=compute_gradient, args=(x_mat_aug, y_vec,
                                                            num_train_ex),
                             avextol=1e-10, epsilon=1e-10, maxiter=400,
                             full_output=1)
    theta_opt = numpy.reshape(f_min_ncg_out[0], (num_features+1, 1), order='F')
    print("theta:")
    print("%s\n" % numpy.array_str(numpy.round(theta_opt, 6)))
    curr_season_mat = point_diff_mat_list['curr_season_mat']
    ones_test_vec = numpy.ones((curr_season_mat.shape[0], 1))
    x_test_mat = curr_season_mat[:, 3]
    x_test_mat_aug = numpy.c_[ones_test_vec, x_test_mat]
    winning_prob = compute_sigmoid(numpy.dot(x_test_mat_aug, theta_opt))
    init_pred_mat = coin_flip(team_ids)
    pred_mat = numpy.c_[init_pred_mat[:, 0:2], winning_prob]

    # Coin-flip algorithm
    # file_name = "coin_flips.csv"
    # pred_mat = coin_flip(team_ids)

    # Generate raw submission file
    # file_name = "sample_submission.csv"
    gen_raw_submission(file_name, pred_mat)

    # Load raw submission file
    print("Loading submission file.")
    raw_submission = numpy.genfromtxt(file_name, dtype=None, delimiter=",")

    # Parse raw submission file
    submission = parse_raw_submission(raw_submission)

    # Load current tournament results
    print("Loading 2014 NCAA Tournament results.")
    tournament_results = numpy.genfromtxt("tourney_results_2014.csv",
                                          delimiter=",")

    # Compute evaluation function of submission
    log_loss = evaluate_submission(tournament_results,submission)
    print("Predicted binomial deviance:  %.5f" % log_loss)

# Call main function
if __name__ == "__main__":
    main()
