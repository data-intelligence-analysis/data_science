import pandas as pd
import numpy as np
import seaborn as sns
import matpotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to monokai theme
# this line of code is important so that we can see the x and y axis clearly
# if you don't run this code line, you will notice that the xlabel and the ylabel on my plot is black on black and it will
             
# Load the data
tweets_df = pd.read_csv('twitter.csv')

#print DataFrame
tweets_df

#tabulated version of the dataFrame
tweets_df.info()

#<class 'pandas.core.frame.DataFrame'>
#RangeIndex: 31962 entries, 0 to 31961
#Data columns (total 3 columns):
# #   Column  Non-Null Count  Dtype 
---  ------  --------------  ----- 
# 0   id      31962 non-null  int64 
# 1   label   31962 non-null  int64 
# 2   tweet   31962 non-null  object
#dtypes: int64(2), object(1)
#memory usage: 749.2+ KB

#Statistical forms of the data
tweet_df.describe()

	#             id	label
#count	31962.000000	31962.000000
#mean	15981.500000	0.070146
#std	9226.778988	0.255397
#min	1.000000	0.000000
#25%	7991.250000	0.000000
#50%	15981.500000	0.000000
#75%	23971.750000	0.000000
#max	31962.000000	1.000000


             
