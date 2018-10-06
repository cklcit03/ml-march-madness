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
# Apply ML methods to predict outcome of an NCAA Tournament
# Function that computes Elo differential feature
import numpy
import itertools


def elo_pred(elo_A, elo_B):
    """ Computes probability of team A defeating team B given their respective
        Elo ratings.

    Args:
      elo_A: Elo rating of team A
      elo_B: Elo rating of team B

    Returns:
      win_prob_A: Probability of team A defeating team B
    """
    elo_diff = elo_A-elo_B
    win_prob_A = (1./(10.**(-elo_diff/400.)+1.))
    return win_prob_A


def expected_margin(elo_diff):
    """ Computes expected margin between teams A and B given their Elo rating
        differential before their matchup.

    Args:
      elo_diff: Elo rating differential between teams A and B before their
                matchup

    Returns:
      exp_margin_A_B: Expected margin between teams A and B
    """
    term1 = 7.5
    term2 = 0.006
    # term1 = 15
    # term2 = -0.007
    exp_margin_A_B = term1+term2*elo_diff
    return exp_margin_A_B


def elo_update(w_elo, l_elo, margin):
    """ Computes update to Elo ratings of winning and losing teams.

    Args:
      w_elo: Elo rating of winning team before matchup
      l_elo: Elo rating of losing team before matchup
      margin: Actual margin between winning and losing teams

    Returns:
      update: Update to Elo ratings of winning and losing teams
    """
    elo_K_factor = 20
    margin_offset = 3.0
    margin_power = 0.8
    # elo_K_factor = 8
    # margin_offset = 3.5
    # margin_power = 1.45
    elo_diff = w_elo-l_elo
    pred = elo_pred(w_elo,l_elo)
    margin_of_victory_mult = (
        ((margin+margin_offset)**margin_power)/expected_margin(elo_diff))
    update = elo_K_factor*margin_of_victory_mult*(1-pred)
    return update


def gen_elo_differential(regular_season_results, team_ids, training_data,
                         test_season_ids, curr_season_id):
    """ Generates matrix of Elo differentials between teams A and B for each
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
      test_season_ids: Array of integers denoting IDs of seasons for test data
      curr_season_id: Integer denoting ID of current season

    Returns:
      return_list: List of three objects.
                   elo_diff_mat: Matrix that consists of these columns:
                                 Column 1: integer denoting season ID
                                 Column 2: integer denoting ID of team A
                                 Column 3: integer denoting ID of team B
                                 (assume that in each row, value in Column 3
                                 exceeds value in Column 2)
                                 Column 4: 0 if team A lost to team B;
                                 otherwise, 1 (assume that A and B played in
                                 that season's tournament)
                                 Column 5: difference between final Elo rating
                                 of team A and final Elo rating of team B for
                                 this season
                   test_season_mat: Matrix that consists of these columns:
                                    Column 1: integer denoting test season ID
                                    Column 2: integer denoting ID of team A
                                    Column 3: integer denoting ID of team B
                                    (assume that in each row, value in Column 3
                                    exceeds value in Column 2)
                                    Column 4: difference between final Elo
                                    rating of team A and final Elo rating of team
                                    B for test season
                   curr_season_mat: Matrix that consists of these columns:
                                    Column 1: integer denoting current season ID
                                    Column 2: integer denoting ID of team A
                                    Column 3: integer denoting ID of team B
                                    (assume that in each row, value in Column 3
                                    exceeds value in Column 2)
                                    Column 4: difference between final Elo
                                    rating of team A and final Elo rating of team
                                    B for current season
    """
    regular_season_results_no_header = regular_season_results[1:, :]
    season_ids = regular_season_results_no_header[:, 0]
    unique_season_ids = numpy.unique(season_ids)
    unique_season_ids_i = numpy.zeros((unique_season_ids.shape[0], 1))
    for season_idx in range(0, unique_season_ids.shape[0]):
        unique_season_ids_i[season_idx] = ord(unique_season_ids[season_idx])
    num_unique_seasons = unique_season_ids.shape[0]
    curr_const_mat = numpy.ones((training_data.shape[0], 2))
    elo_diff_mat = numpy.c_[training_data, curr_const_mat]
    elo_start_rating = 1500
    home_advantage = 100
    num_test_seasons = 0
    for season_idx in range(0, num_unique_seasons):
        season_id = ord(unique_season_ids[season_idx])
        game_indices = numpy.where(season_ids == unique_season_ids[season_idx])
        season_results = regular_season_results_no_header[game_indices[0], :]
        winner_ids = season_results[:, 2].astype(float)
        winner_scores = season_results[:, 3].astype(float)
        loser_ids = season_results[:, 4].astype(float)
        loser_scores = season_results[:, 5].astype(float)

        # For each season and team, compute final Elo rating
        elo_dict = dict(zip(team_ids, [elo_start_rating]*team_ids.shape[0]))
        for game_idx in range(0, season_results.shape[0]):
            curr_winner = winner_ids[game_idx]
            curr_loser = loser_ids[game_idx]
            winner_score = winner_scores[game_idx]
            loser_score = loser_scores[game_idx]
            curr_margin = winner_score-loser_score
            curr_location = season_results[game_idx, 6]

            # Determine home-court advantage
            winner_advantage = 0
            loser_advantage = 0
            if curr_location == "H":
                winner_advantage += home_advantage
            elif curr_location == "A":
                loser_advantage += home_advantage

            # Update Elo ratings
            init_winner_elo = elo_dict[curr_winner]+winner_advantage
            init_loser_elo = elo_dict[curr_loser]+loser_advantage
            curr_update = elo_update(init_winner_elo, init_loser_elo,
                                     curr_margin)
            elo_dict[curr_winner] += curr_update
            elo_dict[curr_loser] -= curr_update

        # For each season, consider all (team A, team B) pairings where teams A
        # and B played each other in the tournament
        # Compute difference between final Elo ratings of teams A and B
        if ((season_id not in test_season_ids) and
            (season_id != curr_season_id)): 
            season_idx = numpy.where((elo_diff_mat[:, 0] == season_id))
            for pair_idx in season_idx[0]:
                idA = elo_diff_mat[pair_idx, 1]
                idB = elo_diff_mat[pair_idx, 2]
                elo_A = elo_dict[idA]
                elo_B = elo_dict[idB]
                elo_diff_mat[pair_idx, 4] = elo_A-elo_B
        elif (season_id in test_season_ids):
            team_ids_list = team_ids.tolist()
            team_id_pairs = itertools.combinations(team_ids_list, 2)
            team_id_pairs_array = numpy.asarray(list(team_id_pairs))
            test_season_mat = numpy.zeros((team_id_pairs_array.shape[0], 5))
            for pair_idx in range(0, team_id_pairs_array.shape[0]):
                idA = team_id_pairs_array[pair_idx, 0]
                idB = team_id_pairs_array[pair_idx, 1]
                elo_A = elo_dict[idA]
                elo_B = elo_dict[idB]
                test_season_mat[pair_idx, 0] = season_id
                test_season_mat[pair_idx, 1] = idA
                test_season_mat[pair_idx, 2] = idB
                test_season_mat[pair_idx, 3] = elo_A-elo_B
                test_season_mat[pair_idx, 4] = elo_pred(elo_A, elo_B)
            num_test_seasons = num_test_seasons+1
            if (num_test_seasons > 1):
                final_test_mat = numpy.r_[final_test_mat, test_season_mat]
            else:
                final_test_mat = test_season_mat
        else:
            team_ids_list = team_ids.tolist()
            team_id_pairs = itertools.combinations(team_ids_list, 2)
            team_id_pairs_array = numpy.asarray(list(team_id_pairs))
            curr_season_mat = numpy.zeros((team_id_pairs_array.shape[0], 5))
            for pair_idx in range(0, team_id_pairs_array.shape[0]):
                idA = team_id_pairs_array[pair_idx, 0]
                idB = team_id_pairs_array[pair_idx, 1]
                elo_A = elo_dict[idA]
                elo_B = elo_dict[idB]
                curr_season_mat[pair_idx, 0] = curr_season_id
                curr_season_mat[pair_idx, 1] = idA
                curr_season_mat[pair_idx, 2] = idB
                curr_season_mat[pair_idx, 3] = elo_A-elo_B
                curr_season_mat[pair_idx, 4] = elo_pred(elo_A, elo_B)
    return_list = {'elo_diff_mat': elo_diff_mat,
                   'test_season_mat': final_test_mat,
                   'curr_season_mat': curr_season_mat}
    return return_list
