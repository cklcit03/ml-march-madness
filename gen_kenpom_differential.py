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
# Function that computes KenPom differential feature
import numpy
import itertools


def gen_kenpom_differential(kenpom_data, team_ids, training_data,
                            curr_season_id, curr_const):
    """ Generates matrix of KenPom differentials between teams A and B for each
        season of interest.

    Args:
      kenpom_data: Matrix that consists of these columns:
                   Column 1: character denoting season ID
                   Column 2: integer denoting ID of team A
                   Column 3: float denoting adjusted efficiency margin (AEM) of
                   team A for that season
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
                   kenpom_diff_mat: Matrix that consists of these columns:
                                    Column 1: integer denoting season ID
                                    Column 2: integer denoting ID of team A
                                    Column 3: integer denoting ID of team B
                                    (assume that in each row, value in Column 3
                                    exceeds value in Column 2)
                                    Column 4: 0 if team A lost to team B;
                                    otherwise, 1 (assume that A and B played in
                                    that season's tournament)
                                    Column 5: difference between AEM of team A
                                    and AEM of team B for this season
                   curr_season_mat: Matrix that consists of these columns:
                                    Column 1: integer denoting current season ID
                                    Column 2: integer denoting ID of team A
                                    Column 3: integer denoting ID of team B
                                    (assume that in each row, value in Column 3
                                    exceeds value in Column 2)
                                    Column 4: difference between AEM of team A
                                    and AEM of team B for current season
    """
    season_ids = kenpom_data[:, 0]
    unique_season_ids = numpy.unique(season_ids)
    unique_season_ids_i = numpy.zeros((unique_season_ids.shape[0], 1))
    for season_idx in range(0, unique_season_ids.shape[0]):
        unique_season_ids_i[season_idx] = ord(unique_season_ids[season_idx])
    num_unique_seasons = unique_season_ids.shape[0]
    curr_const_mat = curr_const*numpy.ones((training_data.shape[0], 1))
    kenpom_diff_mat = numpy.c_[training_data, curr_const_mat]
    for season_idx in range(0, num_unique_seasons):
        season_id = ord(unique_season_ids[season_idx])
        game_indices = numpy.where(season_ids == unique_season_ids[season_idx])
        curr_season_results = kenpom_data[game_indices[0], :]
        curr_season_ids = curr_season_results[:, 1].astype(float)

        # For each season, consider all (team A, team B) pairings where teams A
        # and B played each other in the tournament
        # Compute difference between AEM of teams A and B
        if ((season_id != curr_season_id)): 
            season_idx1 = numpy.where((kenpom_diff_mat[:, 0] == season_id))
            for pair_idx in season_idx1[0]:
                idA = kenpom_diff_mat[pair_idx, 1]
                idB = kenpom_diff_mat[pair_idx, 2]
                idA_idx = numpy.where(curr_season_ids == idA)
                idB_idx = numpy.where(curr_season_ids == idB)
                aemA = curr_season_results[idA_idx[0], 2].astype(float)
                aemB = curr_season_results[idB_idx[0], 2].astype(float)
                kenpom_diff_mat[pair_idx, 4] = aemA-aemB
        else:
            team_ids_list = team_ids.tolist()
            team_id_pairs = itertools.combinations(team_ids_list, 2)
            team_id_pairs_array = numpy.asarray(list(team_id_pairs))
            curr_season_mat = numpy.zeros((team_id_pairs_array.shape[0], 4))
            for pair_idx in range(0, team_id_pairs_array.shape[0]):
                idA = team_id_pairs_array[pair_idx, 0]
                idB = team_id_pairs_array[pair_idx, 1]
                idA_idx = numpy.where(curr_season_ids == idA)
                idB_idx = numpy.where(curr_season_ids == idB)
                aemA = curr_season_results[idA_idx[0], 2].astype(float)
                aemB = curr_season_results[idB_idx[0], 2].astype(float)
                curr_season_mat[pair_idx, 0] = curr_season_id
                curr_season_mat[pair_idx, 1] = idA
                curr_season_mat[pair_idx, 2] = idB
                curr_season_mat[pair_idx, 3] = aemA-aemB
    return_list = {'kenpom_diff_mat': kenpom_diff_mat,
                   'curr_season_mat': curr_season_mat}
    return return_list
