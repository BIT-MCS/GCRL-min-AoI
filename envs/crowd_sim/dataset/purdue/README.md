# mcs-mobility-trace
We conducted a real-world mobile crowdsensing campaign on our university campus (Purdue) over one month (Feb 7 - Mar 7, 2018) with 50 users to evaluate CROWDBIND (our EWSN 20 paper) plus 4 competing solutions. The students went about their daily routine with their smartphones running the five different software packages, collecting sensor data (e.g. pressure). This is the anonymized mobility trace of these 50 students over a 100 sq. km. area. 

If you use this trace, please cite our paper:

Heng Zhang, Michael A. Roth (Google), Rajesh K. Panta (AT&T Labs Research), He Wang, and Saurabh Bagchi, “CrowdBind: Fairness Enhanced Late Binding Task Scheduling in Mobile Crowdsensing,” At the 17th International Conference on Embedded Wireless Systems and Networks (EWSN), pp. 1-12, Feb 17-19, 2020, Lyon, France. (Best paper award winner)


## How the data looks like
This mobility trace is stored as numpy 2-d array. 
* The elements of each row: [('id', 'int'), ('latitude', 'float'), ('longitude', 'float'), ('timestamp', 'int')])
* The first element of each row is the id of a user. For user privacy concern, we use numbers for the 'id' to denote those users. The entreis with the same "id" come from one user.
* The timestamp is in Unix epoch time and you can use Python [datetime module](https://docs.python.org/2/library/datetime.html). Be careful to refer to the module corresponding to the correct Python version.
* Note that some users may have some missing datapoints. This may due to different factors such as user turned off phone at night, user has limited network access, or user quitted the data collection campaign for personal reasons. We did not restrict user's normal activities during the data collection.
