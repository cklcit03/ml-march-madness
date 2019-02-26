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
    teams = numpy.genfromtxt("teams_2018.csv", dtype=str, delimiter=",")
    team_ids = teams[1:, 0]
    num_teams = team_ids.shape[0]
    team_names = teams[1:, 1]
    team_names_list = team_names.tolist()

    # Iterate over seasons (starting in 2001-02)
    num_seasons = 17
    base_year = 2002
    for season_idx in range(0, num_seasons):
        file_mat = 10000*numpy.ones((num_teams, 3), dtype=object)
        file_mat[:, 1] = team_ids
        curr_year = base_year+season_idx
        for team_idx in range(0, num_teams):
            file_mat[team_idx, 0] = curr_year
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
                    school_id = 1157
                elif (school == 'East Tennessee St.'):
                    school_id = 1190
                elif (school == 'FIU'):
                    school_id = 1198
                elif (school == 'Fort Wayne'):
                    school_id = 1236
                elif (school == 'Kent St.'):
                    school_id = 1245
                elif (school == 'LIU Brooklyn'):
                    school_id = 1254
                elif (school == 'Louisiana Lafayette'):
                    school_id = 1418
                elif (school == 'Louisiana Monroe'):
                    school_id = 1419
                elif (school == 'Maryland Eastern Shore'):
                    school_id = 1271
                elif (school == 'Middle Tennessee'):
                    school_id = 1292
                elif (school == 'Mississippi Valley St.'):
                    school_id = 1290
                elif (school == 'Nebraska Omaha'):
                    school_id = 1303
                elif (school == 'North Carolina A&T'):
                    school_id = 1299
                elif (school == 'North Carolina Central'):
                    school_id = 1300
                elif (school == 'North Carolina St.'):
                    school_id = 1301
                elif (school == 'North Dakota St.'):
                    school_id = 1295
                elif (school == 'Northwestern St.'):
                    school_id = 1322
                elif (school == 'South Carolina St.'):
                    school_id = 1354
                elif (school == 'South Dakota St.'):
                    school_id = 1355
                elif (school == 'Southern'):
                    school_id = 1380
                elif (school == 'Southwest Missouri St.'):
                    school_id = 1283
                elif (school == 'Southwest Texas St.'):
                    school_id = 1402
                elif (school == 'Tennessee Martin'):
                    school_id = 1404
                elif (school == 'Texas Pan American'):
                    school_id = 1410
                elif (school == 'UT Rio Grande Valley'):
                    school_id = 1410
                elif (school == 'UMKC'):
                    school_id = 1282
                elif (school == 'UTSA'):
                    school_id = 1427
                elif (school == 'VCU'):
                    school_id = 1433
                elif (school == 'Western Kentucky'):
                    school_id = 1443
                else:
                    found_flag = 1
                    print("school = %s, result = %s" % (school, result))
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
    numpy.savetxt('kenpom_2018.csv', total_file_mat, fmt='%s', delimiter=',')

# Call main function
if __name__ == "__main__":
    main()
