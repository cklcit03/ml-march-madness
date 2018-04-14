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
import numpy
import itertools
import math


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def parse_prev_tourney_results(prev_tourney_results):
    """ Parses matrix of previous tournament results.

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
      parsed_prev_tourney_results: Matrix that consists of these columns:
                                   Column 1: character denoting season ID
                                   Column 2: integer denoting ID of team A
                                   Column 3: integer denoting ID of team B
                                             (assume that in each row, value in
                                             Column 3 exceeds that of value in
                                             Column 2)
                                   Column 4: 0 if team A lost to team B;
                                             otherwise, 1
    """
    num_prev_tourney_games = prev_tourney_results.shape[0]
    parsed_prev_tourney_results = numpy.zeros((num_prev_tourney_games, 4))
    for prev_tourney_game_idx in range(0, num_prev_tourney_games):
        season_id = prev_tourney_results[prev_tourney_game_idx, 0]
        winning_team_id = prev_tourney_results[prev_tourney_game_idx, 2]
        losing_team_id = prev_tourney_results[prev_tourney_game_idx, 4]
        if (winning_team_id < losing_team_id):
            team_A = winning_team_id
            team_B = losing_team_id
            outcome = 1
        else:
            team_A = losing_team_id
            team_B = winning_team_id
            outcome = 0
        parsed_prev_tourney_results[prev_tourney_game_idx, 0] = season_id
        parsed_prev_tourney_results[prev_tourney_game_idx, 1] = team_A
        parsed_prev_tourney_results[prev_tourney_game_idx, 2] = team_B
        parsed_prev_tourney_results[prev_tourney_game_idx, 3] = outcome
    return parsed_prev_tourney_results


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
    team_ids_list.pop(0)
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
    team_ids = teams[:, 0]
    print("Loading regular season results.")
    regular_season_results = numpy.genfromtxt("regular_season_results.csv",
                                              delimiter=",")
    print("Loading previous tournament results.")
    prev_tourney_results = numpy.genfromtxt("tourney_results.csv",
                                            delimiter=",")

    # Parse previous tournament results
    prev_tourney_mat = parse_prev_tourney_results(prev_tourney_results)

    # Generate predictions and raw submission file
    # file_name = "sample_submission.csv"
    file_name = "coin_flips.csv"
    pred_mat = coin_flip(team_ids)
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
