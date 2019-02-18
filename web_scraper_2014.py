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
# Web scraper that obtains additional features
from bs4 import BeautifulSoup
import difflib
import numpy
import urllib2


class Error(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)


def main():
    """ Main function
    """
    print("Loading list of teams.")
    teams = numpy.genfromtxt("teams_2014.csv", dtype=str, delimiter=",")
    team_ids = teams[1:, 0]
    num_teams = team_ids.shape[0]
    team_names = teams[1:, 1]
    team_names_list = team_names.tolist()

    # Iterate over seasons (starting in 2001-02)
    num_seasons = 13
    base_season = ord('G')
    base_year = 2002
    for season_idx in range(0, num_seasons):
        curr_season = base_season+season_idx
        file_mat = 10000*numpy.ones((num_teams, 3), dtype=object)
        file_mat[:, 1] = team_ids
        for team_idx in range(0, num_teams):
            file_mat[team_idx, 0] = chr(curr_season)
        curr_year = base_year+season_idx
        print("season = %d" % curr_year)
        url_string = 'https://kenpom.com/index.php?y=%d' % curr_year
        page = urllib2.urlopen(url_string)
        soup = BeautifulSoup(page, 'html.parser')
        trs = soup.find_all('tr')
        tr_idx = 0
        for tr in trs:
            tds = tr.find_all('td')
            if tds:
                school = tds[1].a.string
                result = difflib.get_close_matches(school, team_names_list)
                found_flag = 0
                if (school == 'Coastal Carolina'):
                    school_id = 553
                elif (school == 'East Tennessee St.'):
                    school_id = 586
                elif (school == 'FIU'):
                    school_id = 594
                elif (school == 'Kent St.'):
                    school_id = 639
                elif (school == 'LIU Brooklyn'):
                    school_id = 648
                elif (school == 'Louisiana Lafayette'):
                    school_id = 811
                elif (school == 'Louisiana Monroe'):
                    school_id = 812
                elif (school == 'Maryland Eastern Shore'):
                    school_id = 665
                elif (school == 'Mississippi Valley St.'):
                    school_id = 685
                elif (school == 'Nebraska Omaha'):
                    school_id = 697
                elif (school == 'North Carolina A&T'):
                    school_id = 693
                elif (school == 'North Carolina Central'):
                    school_id = 694
                elif (school == 'North Carolina St.'):
                    school_id = 695
                elif (school == 'North Dakota St.'):
                    school_id = 689
                elif (school == 'Northwestern St.'):
                    school_id = 716
                elif (school == 'South Carolina St.'):
                    school_id = 747
                elif (school == 'South Dakota St.'):
                    school_id = 748
                elif (school == 'Southern'):
                    school_id = 773
                elif (school == 'Southwest Missouri St.'):
                    school_id = 678
                elif (school == 'Southwest Texas St.'):
                    school_id = 795
                elif (school == 'Tennessee Martin'):
                    school_id = 797
                elif (school == 'UMKC'):
                    school_id = 677
                elif (school == 'UTSA'):
                    school_id = 820
                elif (school == 'VCU'):
                    school_id = 825
                else:
                    found_flag = 1
                    best_match = result[0]
                    match_idx = numpy.where(team_names == best_match)
                    school_id = team_ids[match_idx[0]].astype(int)
                if (found_flag == 0):
                    match_idx = numpy.where(team_ids.astype(int) == school_id)
                file_mat[match_idx[0], 2] = float(tds[4].string)
                tr_idx = tr_idx+1
        print("number of teams = %d" % tr_idx)
        if (season_idx == 0):
            total_file_mat = file_mat
        else:
            total_file_mat = numpy.r_[total_file_mat, file_mat]
    numpy.savetxt('kenpom_2014.csv', total_file_mat, fmt='%s', delimiter=',')

# Call main function
if __name__ == "__main__":
    main()
