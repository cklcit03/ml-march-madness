# Copyright (C) 2019  Caleb Lo
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
# Apply ML methods to predict outcome of 2015 NCAA Tournament
from matplotlib import pyplot
from tempfile import TemporaryFile
from gen_kenpom_differential import gen_kenpom_differential
from gen_srs_differential import gen_srs_differential
from support_vector_machine import train_and_test_svm
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
        training_data[prev_tourney_game_idx, 0] = season_id
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


def gen_raw_submission(file_name, pred_mat, curr_season_id, curr_tourney_teams):
    """ Generates raw submission file.
    Args:
      file_name: Name of raw submission file.
      pred_mat: Matrix of predictions, where first and second columns include
                IDs of distinct teams; each entry in third column is probability
                that team in first column defeats team in second column.  Only
                unordered pairings of teams appear in this matrix.
      curr_season_id: Year of current season
      curr_tourney_teams: IDs of teams in this year's tournament
    Returns:
      None.
    """
    num_pairs = pred_mat.shape[0]
    num_tourney_teams = curr_tourney_teams.shape[0]
    num_tourney_pairs = (num_tourney_teams)*(num_tourney_teams-1)/2
    file_mat = numpy.zeros((num_tourney_pairs+1, 2), dtype=object)
    file_mat[0, 0] = "id"
    file_mat[0, 1] = "pred"
    team_counter = 0
    for pair_idx in range(0, num_pairs):
        id1 = pred_mat[pair_idx, 0].astype(float)
        id2 = pred_mat[pair_idx, 1].astype(float)
        curr_id = "%d_%d_%d" % (curr_season_id, id1, id2)
        if (id1 in curr_tourney_teams and id2 in curr_tourney_teams):
            file_mat[team_counter+1, 0] = curr_id
            file_mat[team_counter+1, 1] = str(pred_mat[pair_idx, 2])
            team_counter = team_counter+1
    numpy.savetxt(file_name, file_mat, fmt='%s', delimiter=',')


def parse_raw_submission(raw_submission):
    """ Parses raw submission input and returns matrix.
    Args:
      raw_submission: Matrix of raw submission input, where first column
                      includes entries of the form "YYYY_A_B" where A and B are
                      the IDs of two teams and YYYY is current season; second
                      column includes probabilities that team A will defeat team
                      B.
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


def evaluate_submission(results, submission, season_ids, bound_extreme):
    """ Computes predicted binomial deviance between results and submission for
        a set of seasons.
    Args:
      results: Matrix of tournament results, where each row represents a game;
               third and fifth columns include IDs of winning and losing teams,
               respectively.
      submission: Matrix of submissions, where each row represents a game; first
                  two columns include IDs of two teams and third column includes
                  probability that first team wins.
      season_ids: Array of integers denoting IDs of seasons for submission
      bound_extreme: Scalar that is used to prevent overflow in computations
    Returns:
      log_loss: Predicted binomial deviance.
    """
    str_seasons = results[1:, 0]
    seasons = numpy.zeros((str_seasons.shape[0], 1))
    for season_index in range(0, str_seasons.shape[0]):
        seasons[season_index] = str_seasons[season_index]
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

    # Compute log loss
    log_loss = (-1/num_games)*numpy.sum(log_loss_array)
    return log_loss


def main():
    """ Main function
    """
    print("Loading list of teams.")
    teams = numpy.genfromtxt("teams_2015.csv", delimiter=",")
    team_ids = teams[1:, 0]
    print("Loading regular season results.")
    regular_season_results = (
        numpy.genfromtxt("regular_season_results_2015.csv",
                         delimiter=","))
    print("Loading tournament results.")
    tourney_results = numpy.genfromtxt("tourney_results_prev_2015.csv",
                                       delimiter=",")
    print("Loading current tournament results.")
    curr_tourney_results = numpy.genfromtxt("tourney_results_2015.csv",
                                            delimiter=",")
    print("Loading KenPom data.")
    kenpom_data = numpy.genfromtxt("kenpom_2015.csv", delimiter=",")

    # Generate training results
    training_mat = gen_train_results(tourney_results)

    # Initialize parameters
    bound_extreme = 0.0000000000000020278
    curr_const = 0.001
    curr_season_id = 2015
    feat_norm_flag = 0
    num_splits = 10
    train_ratio = 0.75
    year_flag = 1

    # Compute SRS differential between teams A and B for each season
    print("Computing SRS differential...")
    srs_diff_mat_list = gen_srs_differential(regular_season_results,
                                             team_ids, training_mat,
                                             curr_season_id, curr_const,
                                             year_flag)
    srs_diff_mat = srs_diff_mat_list['srs_diff_mat']

    # Set training features and labels
    srs_idx1 = numpy.where(srs_diff_mat[:, 4] != curr_const)
    srs_idx = srs_idx1[0]
    if (feat_norm_flag == 1):
        feature_srs_scale = feature_normalize(srs_diff_mat[srs_idx, 4])
    else:
        feature_srs_scale = srs_diff_mat[srs_idx, 4]
    x_mat = numpy.reshape(feature_srs_scale, (feature_srs_scale.shape[0], 1))
    label_vec = srs_diff_mat[srs_idx, 3]

    # Set current season testing features and labels
    srs_diff_curr_season = srs_diff_mat_list['curr_season_mat']
    if (feat_norm_flag == 1):
        curr_feature_srs_scale = feature_normalize(srs_diff_curr_season[:, 3])
    else:
        curr_feature_srs_scale = srs_diff_curr_season[:, 3]
    x_curr_mat = numpy.reshape(curr_feature_srs_scale,
                               (curr_feature_srs_scale.shape[0], 1))

    # Compute KenPom differential between teams A and B for each season
    print("Computing KenPom differential...")
    kenpom_diff_mat_list = gen_kenpom_differential(kenpom_data, team_ids,
                                                   training_mat,
                                                   curr_season_id, curr_const,
                                                   year_flag)
    kenpom_diff_mat = kenpom_diff_mat_list['kenpom_diff_mat']
    kenpom_idx1 = numpy.where(kenpom_diff_mat[:, 4] != curr_const)
    kenpom_idx = kenpom_idx1[0]
    if (feat_norm_flag == 1):
        feature_kenpom_scale = feature_normalize(kenpom_diff_mat[kenpom_idx, 4])
    else:
        feature_kenpom_scale = kenpom_diff_mat[kenpom_idx, 4]
    kenpom_x_mat = numpy.reshape(feature_kenpom_scale,
                                 (feature_kenpom_scale.shape[0], 1))
    kenpom_diff_curr_season = kenpom_diff_mat_list['curr_season_mat']
    if (feat_norm_flag == 1):
        curr_feature_kenpom_scale = (
            feature_normalize(kenpom_diff_curr_season[:, 3]))
    else:
        curr_feature_kenpom_scale = kenpom_diff_curr_season[:, 3]
    kenpom_x_curr_mat = numpy.reshape(curr_feature_kenpom_scale,
                                      (curr_feature_kenpom_scale.shape[0], 1))
    kenpom_label_vec = kenpom_diff_mat[kenpom_idx, 3]
    num_kenpom_games = kenpom_x_mat.shape[0]
    num_games = x_mat.shape[0]
    x_comb_mat = numpy.c_[x_mat[(num_games-num_kenpom_games):num_games, :],
                          kenpom_x_mat]
    x_curr_mat = numpy.c_[x_curr_mat, kenpom_x_curr_mat]

    # Randomize train/test splits
    numpy.random.seed(10)
    test_array = numpy.zeros((num_splits, 1))
    init_x_curr_mat = x_curr_mat
    for split_idx in range(0, num_splits):
        # print("")
        train_idx = numpy.random.choice(x_comb_mat.shape[0],
                                        train_ratio*x_comb_mat.shape[0],
                                        replace=False)
        test_idx = numpy.setdiff1d(numpy.arange(x_comb_mat.shape[0]), train_idx)
        train_mat = x_comb_mat[train_idx, :]
        train_label = kenpom_label_vec[train_idx]
        x_test_mat = x_comb_mat[test_idx, :]
        test_label = kenpom_label_vec[test_idx]

        # Use SVM for training and testing
        svm_list = train_and_test_svm(train_mat, train_label, x_test_mat,
                                      x_curr_mat)
        test_prob = svm_list['test_prob']
        curr_prob = svm_list['curr_prob']

        # Compute evaluation function of test data
        num_test_games = test_label.shape[0]
        log_loss_array = numpy.zeros((num_test_games, 1))
        for game_index in range(0, num_test_games):
            curr_outcome = test_label[game_index]
            curr_game_prob = test_prob[game_index]

            # Bound prediction from extremes if necessary
            if (curr_game_prob == 1):
                curr_game_prob = 1-bound_extreme
            elif (curr_game_prob == 0):
                curr_game_prob = bound_extreme

            # Evaluate per-game log loss
            log_loss_term1 = curr_outcome*math.log(curr_game_prob)
            log_loss_term2 = (1-curr_outcome)*math.log(1-curr_game_prob)
            log_loss_array[game_index] = log_loss_term1+log_loss_term2
        test_log_loss = (-1/num_test_games)*numpy.sum(log_loss_array)
        test_array[split_idx] = test_log_loss

        # Aggregate probabilities for current season
        if (split_idx == 0):
            curr_array = curr_prob
        else:
            curr_array = numpy.c_[curr_array, curr_prob]
    print("")
    print("Average test binomial deviance = %.5f" % numpy.mean(test_array))

    # Compute weighted probabilities for current season
    print("")
    curr_avg_prob = numpy.mean(curr_array, axis=1)

    # Generate raw submission file
    curr_file_name = "curr_submission_2015.csv"
    init_pred_mat = coin_flip(team_ids)
    curr_pred_mat = numpy.c_[init_pred_mat[:, 0:2], curr_avg_prob]
    curr_tourney_winners = curr_tourney_results[1:, 2].astype(int)
    curr_tourney_losers = curr_tourney_results[1:, 4].astype(int)
    curr_tourney_teams = numpy.union1d(curr_tourney_winners,
                                       curr_tourney_losers)
    curr_first_four_losers = numpy.array([1129, 1140, 1264, 1316])
    curr_tourney_teams = numpy.union1d(curr_tourney_teams,
                                       curr_first_four_losers)
    gen_raw_submission(curr_file_name, curr_pred_mat, curr_season_id,
                       curr_tourney_teams)

    # Load raw submission file
    print("Loading curr submission file.")
    curr_raw_submission = numpy.genfromtxt(curr_file_name, dtype=None,
                                           delimiter=",")

    # Parse raw submission file
    curr_submission = parse_raw_submission(curr_raw_submission)

    # Compute evaluation function of submission
    curr_log_loss = evaluate_submission(curr_tourney_results, curr_submission,
                                        [curr_season_id], bound_extreme)
    print("Current binomial deviance = %.5f" % curr_log_loss)

# Call main function
if __name__ == "__main__":
    main()
