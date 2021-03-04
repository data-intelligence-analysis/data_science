import pandas as pd
import numpy as np
import seaborn as sns
import matpotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to monokai theme
# this line of code is important so that we can see the x and y axis clearly
# if you don't run this code line, you will notice that the xlabel and the ylabel on my plot is black on black and it will

## Import libraries and dataset 
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

	#        id	label
#count	31962.000000	31962.000000
#mean	15981.500000	0.070146
#std	9226.778988	0.255397
#min	1.000000	0.000000
#25%	7991.250000	0.000000
#50%	15981.500000	0.000000
#75%	23971.750000	0.000000
#max	31962.000000	1.000000

#display the tweets in the tweet column
tweet_df['tweet']

#display the id in the id column
tweet_df['id']

#drop the 'id' column from the DataFrame. 
#ensoure that the column has been successfully dropped.
tweets_df = tweets_df.drop(['id'], axis = 1) #axis = 1 means you drop the whole column
#print out new dataFrame
tweets_df

#	label	tweet
#0	0	@user when a father is dysfunctional and is s...
#1	0	@user @user thanks for #lyft credit i can't us...
#2	0	bihday your majesty
#3	0	#model i love u take with u all the time in ...
#4	0	factsguide: society now #motivation
#...	...	...
#31957	0	ate @user isz that youuu?Ã°??ÂÃ°??ÂÃ°??ÂÃ°??ÂÃ°??ÂÃ°...
#31958	0	to see nina turner on the airwaves trying to...
#1959	0	listening to sad songs on a monday morning otw...
#31960	1	@user #sikh #temple vandalised in in #calgary,...
#31961	0	thank you @user for you follow


## Task #3: Explore Dataset 

