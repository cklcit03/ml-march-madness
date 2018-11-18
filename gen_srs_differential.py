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
# Function that computes SRS differential feature
import numpy
import scipy
import itertools


def games_and_margin_matrices(curr_season_results, team_ids):
    """ Creates two matrices:
            1) Matrix where each row represents a game in current season.  Each
            column represents a team.  The (i,j)-th entry is 1) 1 if team $j$
            won game $i$, 2) -1 if team $j$ lost game $i$, or 3) 0.  There is an
            additional column; the $i$-th entry in that column is 1) 1 if the
            winner of game $i$ was the home team or 2) -1.
            2) Matrix where each row represents a game in current season.  The
            $i$-th entry is the margin of victory for game $i$.
    Args:
      curr_season_results: Matrix of current-season results that consists of
                           these columns:
                           Column 1: character denoting season ID
                           Column 2: integer denoting ID of date of game
                           Column 3: integer denoting ID of winning team
                           Column 4: integer denoting score of winning team
                           Column 5: integer denoting ID of losing team
                           Column 6: integer denoting score of losing team
                           Column 7: character denoting location of winning team
                           Column 8: number of overtime periods
      team_ids: Vector of team IDs.
    Returns:
      return_list: List of two objects.
                   games_mat: The first matrix that is described in this header.
                   margin_mat: The second matrix that is described in this
                               header.
    """
    num_games = curr_season_results.shape[0]
    num_teams = team_ids.shape[0]
    id_offset = team_ids[0]
    games_mat = numpy.zeros((num_games, num_teams+1))
    margin_mat = numpy.zeros((num_games, 1))
    winners = curr_season_results[:, 2].astype(float)
    losers = curr_season_results[:, 4].astype(float)
    winning_scores = curr_season_results[:, 3].astype(float)
    losing_scores = curr_season_results[:, 5].astype(float)
    for game_idx in range(0, num_games):
        curr_winner = winners[game_idx]
        curr_loser = losers[game_idx]
        curr_location = curr_season_results[game_idx, 6]
        games_mat[game_idx, curr_winner-id_offset] = 1
        games_mat[game_idx, curr_loser-id_offset] = -1
        if (curr_location == b'H'):
            games_mat[game_idx, num_teams] = 1
        elif (curr_location == b'A'):
            games_mat[game_idx, num_teams] = -1
        curr_win_score = winning_scores[game_idx]
        curr_loss_score = losing_scores[game_idx]
        margin_mat[game_idx] = curr_win_score-curr_loss_score
    return_list = {'games_mat': games_mat,
                   'margin_mat': margin_mat}
    return return_list


def gen_srs_differential(regular_season_results, team_ids, training_data,
                         curr_season_id, curr_const):
    """ Generates matrix of SRS differentials between teams A and B for each
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
                   srs_diff_mat: Matrix that consists of these columns:
                                 Column 1: integer denoting season ID
                                 Column 2: integer denoting ID of team A
                                 Column 3: integer denoting ID of team B
                                 (assume that in each row, value in Column 3
                                 exceeds value in Column 2)
                                 Column 4: 0 if team A lost to team B;
                                 otherwise, 1 (assume that A and B played in
                                 that season's tournament)
                                 Column 5: difference between SRS of team A and
                                 SRS of team B for this season
                   curr_season_mat: Matrix that consists of these columns:
                                    Column 1: integer denoting current season ID
                                    Column 2: integer denoting ID of team A
                                    Column 3: integer denoting ID of team B
                                    (assume that in each row, value in Column 3
                                    exceeds value in Column 2)
                                    Column 4: difference between SRS of team A
                                    and SRS of team B for current season
    """
    regular_season_results_no_header = regular_season_results[1:, :]
    season_ids = regular_season_results_no_header[:, 0]
    unique_season_ids = numpy.unique(season_ids)
    unique_season_ids_i = numpy.zeros((unique_season_ids.shape[0], 1))
    for season_idx in range(0, unique_season_ids.shape[0]):
        unique_season_ids_i[season_idx] = ord(unique_season_ids[season_idx])
    num_unique_seasons = unique_season_ids.shape[0]
    curr_const_mat = curr_const*numpy.ones((training_data.shape[0], 1))
    srs_diff_mat = numpy.c_[training_data, curr_const_mat]
    for season_idx in range(0, num_unique_seasons):
        season_id = ord(unique_season_ids[season_idx])
        game_indices = numpy.where(season_ids == unique_season_ids[season_idx])
        season_results = regular_season_results_no_header[game_indices[0], :]

        # For each season and team, compute SRS
        games_margin_mat_list = games_and_margin_matrices(season_results,
                                                          team_ids)
        games_mat = games_margin_mat_list['games_mat']
        margin_mat = games_margin_mat_list['margin_mat']
        ratings_mat = scipy.linalg.lstsq(games_mat, margin_mat)[0]

        # For each season, consider all (team A, team B) pairings where teams A
        # and B played each other in the tournament
        # Compute difference between RPI of teams A and B
        if ((season_id != curr_season_id)): 
            season_idx = numpy.where((srs_diff_mat[:, 0] == season_id))
            for pair_idx in season_idx[0]:
                idA = srs_diff_mat[pair_idx, 1]
                idB = srs_diff_mat[pair_idx, 2]
                idA_idx = numpy.where(team_ids == idA)
                idB_idx = numpy.where(team_ids == idB)
                srsA = ratings_mat[idA_idx[0]]
                srsB = ratings_mat[idB_idx[0]]
                srs_diff_mat[pair_idx, 4] = srsA-srsB
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
                srsA = ratings_mat[idA_idx[0]]
                srsB = ratings_mat[idB_idx[0]]
                curr_season_mat[pair_idx, 0] = curr_season_id
                curr_season_mat[pair_idx, 1] = idA
                curr_season_mat[pair_idx, 2] = idB
                curr_season_mat[pair_idx, 3] = srsA-srsB
    return_list = {'srs_diff_mat': srs_diff_mat,
                   'curr_season_mat': curr_season_mat}
    return return_list
