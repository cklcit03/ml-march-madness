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
# Function that computes point differential feature
import numpy
import itertools


def gen_point_differential(regular_season_results, team_ids, training_data,
                           curr_season_id, curr_const):
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
      curr_season_id: Integer denoting ID of current season
      curr_const: Float denoting dummy value

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
    regular_season_results_no_header = regular_season_results[1:, :]
    season_ids = regular_season_results_no_header[:, 0]
    unique_season_ids = numpy.unique(season_ids)
    unique_season_ids_i = numpy.zeros((unique_season_ids.shape[0], 1))
    for season_idx in range(0, unique_season_ids.shape[0]):
        unique_season_ids_i[season_idx] = ord(unique_season_ids[season_idx])
    num_unique_seasons = unique_season_ids.shape[0]
    curr_const_mat = curr_const*numpy.ones((training_data.shape[0], 1))
    point_diff_mat = numpy.c_[training_data, curr_const_mat]
    for season_idx in range(0, num_unique_seasons):
        season_id = ord(unique_season_ids[season_idx])
        game_indices = numpy.where(season_ids == unique_season_ids[season_idx])
        season_results = regular_season_results_no_header[game_indices[0], :]
        game_days = season_results[:, 1].astype(float)
        winner_ids = season_results[:, 2].astype(float)
        winner_scores = season_results[:, 3].astype(float)
        loser_ids = season_results[:, 4].astype(float)
        loser_scores = season_results[:, 5].astype(float)
        locations = season_results[:, 6]

        # For each season and team, compute point differential
        net_differential = curr_const*numpy.ones((team_ids.shape[0], 1))
        for team_idx in range(0, team_ids.shape[0]):
            curr_team = team_ids[team_idx].astype(float)
            win_indices = numpy.where(winner_ids == curr_team)
            loss_indices = numpy.where(loser_ids == curr_team)

            # First, consider all wins
            total_win_diff = 0
            if (len(win_indices[0]) > 0):
                curr_winner_scores = winner_scores[win_indices[0]]
                curr_loser_scores = loser_scores[win_indices[0]]
                win_diff = numpy.subtract(curr_winner_scores, curr_loser_scores)
                total_win_diff = numpy.sum(win_diff)

            # Then, consider all losses
            total_loss_diff = 0
            if (len(loss_indices[0]) > 0):
                curr_winner_scores = winner_scores[loss_indices[0]]
                curr_loser_scores = loser_scores[loss_indices[0]]
                loss_diff = numpy.subtract(curr_winner_scores,
                                           curr_loser_scores)
                total_loss_diff = numpy.sum(loss_diff)

            # Now compute net differential
            if (len(win_indices[0]) > 0 or len(loss_indices[0]) > 0):
                net_differential[team_idx] = total_win_diff-total_loss_diff

        # For each season, consider all (team A, team B) pairings where teams A
        # and B played each other in the tournament
        # Compute difference between point differentials of teams A and B
        if (season_id != curr_season_id): 
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
                curr_season_mat[pair_idx, 0] = curr_season_id
                curr_season_mat[pair_idx, 1] = idA
                curr_season_mat[pair_idx, 2] = idB
                curr_season_mat[pair_idx, 3] = net_diffA-net_diffB
    return_list = {'point_diff_mat': point_diff_mat,
                   'curr_season_mat': curr_season_mat}
    return return_list
