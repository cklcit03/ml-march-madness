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
# Function that computes seed difference feature
import numpy
import itertools


def gen_seed_difference(tourney_seeds, team_ids, training_data,
                        curr_season_id, curr_const):
    """ Generates matrix of seed differences between teams A and B for each
        season of interest.

    Args:
      tourney_seeds: Matrix of tournament seeds that consists of these columns:
                     Column 1: character denoting season ID
                     Column 2: string denoting seed of team with format 'Ax'
                               where 'A' denotes region and 'x' denotes seed
                               number
                     Column 3: integer denoting ID of team
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
                   seed_diff_mat:  Matrix that consists of these columns:
                                   Column 1: integer denoting season ID
                                   Column 2: integer denoting ID of team A
                                   Column 3: integer denoting ID of team B
                                   (assume that in each row, value in Column 3
                                   exceeds value in Column 2)
                                   Column 4: 0 if team A lost to team B;
                                   otherwise, 1 (assume that A and B played in
                                   that season's tournament)
                                   Column 5: difference between seed of team A
                                   and seed of team B for this season
                   curr_season_mat: Matrix that consists of these columns:
                                    Column 1: integer denoting current season ID
                                    Column 2: integer denoting ID of team A
                                    Column 3: integer denoting ID of team B
                                    (assume that in each row, value in Column 3
                                    exceeds value in Column 2)
                                    Column 4: difference between seed of team A
                                    and seed of team B for current season
    """
    tourney_seeds_no_header = tourney_seeds[1:, :]
    season_ids = tourney_seeds_no_header[:, 0]
    unique_season_ids = numpy.unique(season_ids)
    num_unique_seasons = unique_season_ids.shape[0]
    curr_const_vec = curr_const*numpy.ones((training_data.shape[0], 1))
    seed_diff_mat = numpy.c_[training_data, curr_const_vec]
    for season_idx in range(0, num_unique_seasons):
        season_id = ord(unique_season_ids[season_idx])
        season_indices = numpy.where(season_ids ==
                                     unique_season_ids[season_idx])
        season_results = tourney_seeds_no_header[season_indices[0], :]

        # For each season, find seed for each team
        tourney_ids = season_results[:, 2].astype(float)
        seed_number = curr_const*numpy.ones((team_ids.shape[0], 1))
        for team_idx in range(0, team_ids.shape[0]):
            curr_team = team_ids[team_idx].astype(float)
            tourney_id = numpy.where(tourney_ids == curr_team)
            if len(tourney_id[0]) > 0:
                curr_seed_tmp = season_results[tourney_id[0], 1]
                curr_seed_str = str(curr_seed_tmp[0])
                curr_seed_strip = curr_seed_str[1:3]
                curr_seed = int(curr_seed_strip)
                seed_number[team_idx] = curr_seed

        # For each season, consider all (team A, team B) pairings where teams A
        # and B played each other in the tournament
        # Compute difference between seeds of teams A and B
        if (season_id != curr_season_id): 
            season_idx = numpy.where((seed_diff_mat[:, 0] == season_id))
            for pair_idx in season_idx[0]:
                idA = seed_diff_mat[pair_idx, 1]
                idB = seed_diff_mat[pair_idx, 2]
                idA_idx = numpy.where(team_ids == idA)
                idB_idx = numpy.where(team_ids == idB)
                seed_numA = seed_number[idA_idx[0]]
                seed_numB = seed_number[idB_idx[0]]
                if (seed_numA != curr_const) and (seed_numB != curr_const):
                    seed_diff_mat[pair_idx, 4] = seed_numA-seed_numB
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
                seed_numA = seed_number[idA_idx[0]]
                seed_numB = seed_number[idB_idx[0]]
                curr_season_mat[pair_idx, 0] = curr_season_id
                curr_season_mat[pair_idx, 1] = idA
                curr_season_mat[pair_idx, 2] = idB
                curr_season_mat[pair_idx, 3] = seed_numA-seed_numB
    return_list = {'seed_diff_mat': seed_diff_mat,
                   'curr_season_mat': curr_season_mat}
    return return_list
