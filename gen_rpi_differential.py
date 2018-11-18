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
# Function that computes RPI differential feature
import numpy
import itertools


def win_pct(curr_season_results, team_id, exclude_team_id):
    """ Computes winning percentage for a given team in a given season.
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
      team_id: ID of given team
      exclude_team_id: ID of team to exclude from this calculation (set to -1 if
                       no team is to be excluded in this case)
    Returns:
      win_pct: Winning percentage for a given team in a given season
    """
    winner_ids = curr_season_results[:, 2].astype(float)
    loser_ids = curr_season_results[:, 4].astype(float)
    winner_indices = numpy.where(winner_ids == team_id)
    loser_indices = numpy.where(loser_ids == team_id)
    game_indices = numpy.union1d(winner_indices[0], loser_indices[0])
    team_season_results = curr_season_results[game_indices, :]
    if (exclude_team_id == -1):
        curr_winner_ids = team_season_results[:, 2].astype(float)
        curr_loser_ids = team_season_results[:, 4].astype(float)
        exclude_winner_indices = numpy.where(curr_winner_ids != exclude_team_id)
        exclude_loser_indices = numpy.where(curr_loser_ids != exclude_team_id)
        exclude_indices = numpy.intersect1d(exclude_winner_indices[0],
                                            exclude_loser_indices[0])
        exclude_results = team_season_results[exclude_indices, :]
        num_games = exclude_results.shape[0]
        if (num_games == 0):
            win_pct = 0.5
        else:
            curr_winners = exclude_results[:, 2].astype(float)
            curr_winner_indices = numpy.where(curr_winners == team_id)
            num_wins = curr_winner_indices[0].shape[0]
            win_pct = num_wins/num_games
    else:
        curr_locations = team_season_results[:, 6]
        home_indices = numpy.where(curr_locations == b'H')
        away_indices = numpy.where(curr_locations == b'A')
        neutral_indices = numpy.where(curr_locations == b'N')
        curr_winners = team_season_results[:, 2].astype(float)
        curr_winner_indices = numpy.where(curr_winners == team_id)
        away_wins = numpy.intersect1d(curr_winner_indices[0], away_indices[0])
        num_away_wins = away_wins.shape[0]
        home_wins = numpy.intersect1d(curr_winner_indices[0], home_indices[0])
        num_home_wins = home_wins.shape[0]
        neutral_wins = numpy.intersect1d(curr_winner_indices[0],
                                         neutral_indices[0])
        num_neutral_wins = neutral_wins.shape[0]
        num_weighted_wins = 1.4*num_away_wins+0.6*num_home_wins+num_neutral_wins
        curr_losers = team_season_results[:, 4].astype(float)
        curr_loser_indices = numpy.where(curr_losers == team_id)
        away_losses = numpy.intersect1d(curr_loser_indices[0], away_indices[0])
        num_away_losses = away_losses.shape[0]
        home_losses = numpy.intersect1d(curr_loser_indices[0], home_indices[0])
        num_home_losses = home_losses.shape[0]
        neutral_losses = numpy.intersect1d(curr_loser_indices[0],
                                           neutral_indices[0])
        num_neutral_losses = neutral_losses.shape[0]
        num_weighted_losses = (
            1.4*num_home_losses+0.6*num_away_losses+num_neutral_losses)
        if (num_weighted_wins > 0 or num_weighted_losses > 0):
            win_pct = num_weighted_wins/(num_weighted_wins+num_weighted_losses)
        else:
            win_pct = 0.5
    return win_pct


def opp_win_pct(curr_season_results, team_id):
    """ Computes opponents' winning percentage for a given team in a given
        season.
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
      team_id: ID of given team
    Returns:
      opp_win_pct: Opponents' mean winning percentage for a given team in a
                   given season
    """
    winner_ids = curr_season_results[:, 2].astype(float)
    loser_ids = curr_season_results[:, 4].astype(float)
    winner_indices = numpy.where(winner_ids == team_id)
    loser_indices = numpy.where(loser_ids == team_id)
    game_indices = numpy.union1d(winner_indices[0], loser_indices[0])
    team_season_results = curr_season_results[game_indices, :]
    curr_winner_ids = team_season_results[:, 2].astype(float)
    opponent_winner_indices = numpy.where(curr_winner_ids != team_id)
    opponent_winner_ids = curr_winner_ids[opponent_winner_indices[0]]
    curr_loser_ids = team_season_results[:, 4].astype(float)
    opponent_loser_indices = numpy.where(curr_loser_ids != team_id)
    opponent_loser_ids = curr_loser_ids[opponent_loser_indices[0]]
    opponent_ids = numpy.union1d(opponent_winner_ids, opponent_loser_ids)
    num_opponents = opponent_ids.shape[0]
    if (num_opponents > 0):
        opponent_win_pct = numpy.zeros((num_opponents, 1))
        for opp_idx in range(0, num_opponents):
            curr_opponent = opponent_ids[opp_idx]
            opponent_win_pct[opp_idx] = win_pct(curr_season_results,
                                                curr_opponent, team_id)
        opp_win_pct = numpy.mean(opponent_win_pct)
    else:
        opp_win_pct = 0.5
    return opp_win_pct


def opp_opp_win_pct(curr_season_results, team_id):
    """ Computes opponents' opponents' winning percentage for a given team in a
        given season.
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
      team_id: ID of given team
    Returns:
      opp_opp_win_pct: Opponents' opponents' mean winning percentage for a given
                       team in a given season
    """
    winner_ids = curr_season_results[:, 2].astype(float)
    loser_ids = curr_season_results[:, 4].astype(float)
    winner_indices = numpy.where(winner_ids == team_id)
    loser_indices = numpy.where(loser_ids == team_id)
    game_indices = numpy.union1d(winner_indices[0], loser_indices[0])
    team_season_results = curr_season_results[game_indices, :]
    curr_winner_ids = team_season_results[:, 2].astype(float)
    opponent_winner_indices = numpy.where(curr_winner_ids != team_id)
    opponent_winner_ids = curr_winner_ids[opponent_winner_indices[0]]
    curr_loser_ids = team_season_results[:, 4].astype(float)
    opponent_loser_indices = numpy.where(curr_loser_ids != team_id)
    opponent_loser_ids = curr_loser_ids[opponent_loser_indices[0]]
    opponent_ids = numpy.union1d(opponent_winner_ids, opponent_loser_ids)
    num_opponents = opponent_ids.shape[0]
    opponent_opponent_win_pct = numpy.zeros((num_opponents, 1))
    if (num_opponents > 0):
        for opp_idx in range(0, num_opponents):
            curr_opponent = opponent_ids[opp_idx]
            opponent_opponent_win_pct[opp_idx] = (
                opp_win_pct(curr_season_results, curr_opponent))
        opp_opp_win_pct = numpy.mean(opponent_opponent_win_pct)
    else:
        opp_opp_win_pct = 0.5
    return opp_opp_win_pct


def gen_rpi_differential(regular_season_results, team_ids, training_data,
                         curr_season_id, curr_const):
    """ Generates matrix of RPI differentials between teams A and B for each
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
                   rpi_diff_mat: Matrix that consists of these columns:
                                 Column 1: integer denoting season ID
                                 Column 2: integer denoting ID of team A
                                 Column 3: integer denoting ID of team B
                                 (assume that in each row, value in Column 3
                                 exceeds value in Column 2)
                                 Column 4: 0 if team A lost to team B;
                                 otherwise, 1 (assume that A and B played in
                                 that season's tournament)
                                 Column 5: difference between RPI of team A and
                                 RPI of team B for this season
                   curr_season_mat: Matrix that consists of these columns:
                                    Column 1: integer denoting current season ID
                                    Column 2: integer denoting ID of team A
                                    Column 3: integer denoting ID of team B
                                    (assume that in each row, value in Column 3
                                    exceeds value in Column 2)
                                    Column 4: difference between RPI of team A
                                    and RPI of team B for current season
    """
    regular_season_results_no_header = regular_season_results[1:, :]
    season_ids = regular_season_results_no_header[:, 0]
    unique_season_ids = numpy.unique(season_ids)
    unique_season_ids_i = numpy.zeros((unique_season_ids.shape[0], 1))
    for season_idx in range(0, unique_season_ids.shape[0]):
        unique_season_ids_i[season_idx] = ord(unique_season_ids[season_idx])
    num_unique_seasons = unique_season_ids.shape[0]
    curr_const_mat = curr_const*numpy.ones((training_data.shape[0], 1))
    rpi_diff_mat = numpy.c_[training_data, curr_const_mat]
    for season_idx in range(0, num_unique_seasons):
        season_id = ord(unique_season_ids[season_idx])
        game_indices = numpy.where(season_ids == unique_season_ids[season_idx])
        season_results = regular_season_results_no_header[game_indices[0], :]

        # For each season and team, compute RPI
        rpi = curr_const*numpy.ones((team_ids.shape[0], 1))
        for team_idx in range(0, team_ids.shape[0]):
            curr_team = team_ids[team_idx].astype(float)
            curr_wpct = win_pct(season_results, curr_team, -1)
            curr_opp_wpct = opp_win_pct(season_results, curr_team)
            curr_opp_opp_wpct = opp_opp_win_pct(season_results, curr_team)
            rpi[team_idx] = (
                0.25*curr_wpct+0.5*curr_opp_wpct+0.25*curr_opp_opp_wpct)

        # For each season, consider all (team A, team B) pairings where teams A
        # and B played each other in the tournament
        # Compute difference between RPI of teams A and B
        if (season_id != curr_season_id): 
            season_idx = numpy.where((rpi_diff_mat[:, 0] == season_id))
            for pair_idx in season_idx[0]:
                idA = rpi_diff_mat[pair_idx, 1]
                idB = rpi_diff_mat[pair_idx, 2]
                idA_idx = numpy.where(team_ids == idA)
                idB_idx = numpy.where(team_ids == idB)
                rpiA = rpi[idA_idx[0]]
                rpiB = rpi[idB_idx[0]]
                if (rpiA != curr_const) and (rpiB != curr_const):
                    rpi_diff_mat[pair_idx, 4] = rpiA-rpiB
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
                rpiA = rpi[idA_idx[0]]
                rpiB = rpi[idB_idx[0]]
                curr_season_mat[pair_idx, 0] = curr_season_id
                curr_season_mat[pair_idx, 1] = idA
                curr_season_mat[pair_idx, 2] = idB
                curr_season_mat[pair_idx, 3] = rpiA-rpiB
    return_list = {'rpi_diff_mat': rpi_diff_mat,
                   'curr_season_mat': curr_season_mat}
    return return_list
