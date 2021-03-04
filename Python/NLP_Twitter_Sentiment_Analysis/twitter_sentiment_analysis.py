import pandas as pd
import numpy as np
import seaborn as sns
import matpotlib.pyplot as plt
from jupyterthemes import jtplot
jtplot.style(theme='monokai', context='notebook', ticks=True, grid=False) 
# setting the style of the notebook to monokai theme
# this line of code is important so that we can see the x and y axis clearly
# if you don't run this code line, you will notice that the xlabel and the ylabel on my plot is black on black and it will

# data source: https://www.kaggle.com/arkhoshghalb/twitter-sentiment-analysis-hatred-speech


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
#31962 tweets = 2 columns

## Task #3: Explore Dataset 
#Using seaboard to plot heatmap to check and remove null values - the method / function isnull() checks for null values and returns true if present and false if not
sns.heatmap(tweets_df.isnull(), yticklables = False, cbar = False, cmap="Blues")
#There are no null values

#plot histogram
tweets_df.hist(bins = 30, figsize = (13,5), color = 'r')

# https://seaborn.pydata.org/generated/seaborn.countplot.html
# https://matplotlib.org/2.0.2/api/colors_api.html
#plot a similar figure using seaborn countplot
#sns.countplot(x="label", data=tweets_df, color='red')
label_data = twitter_df["label"]
sns.countplot(label_data, label='Count') # or sns.countplot(twitter_df["label"], label='Count')
#color palette change
sns.countplot(twitter_df['label'], label = 'Count', palette="Set3")

#Create a length column to get the length of the tweets
tweets_df['length'] = tweets_df['tweet'].apply(len)

#create a histogram plot to view distribution of the length of the tweets
tweets_df['length'].plot(bins=100, kind='hist')

#get statistical description of the new dataframe
tweets_df.describe()


#           label	length
#count	31962.000000	31962.000000
#mean	0.070146	84.739628
#std	0.255397	29.455749
#min	0.000000	11.000000
#25%	0.000000	63.000000
#50%	0.000000	88.000000
#75%	0.000000	108.000000
#max	1.000000	274.000000

#Load the smallest message
tweets_df[tweets_df['length] == 11]['tweet'].iloc[0]
#i love you 
tweets_df[tweets_df['length'] == 85]['tweet'].iloc[0]
#' Ã¢\x86\x9d #under the spell of brexit referendum - commerzbank   #blog #silver #gold #forex'		    

#building the NLP divide the data set into two dataframes		    
#analyzing positive sentiments
positive = tweets_df[tweets_df['label'] == 0]
		    
#print out positive tweets
positive

     label	                                          tweet	             length
#0	0	@user when a father is dysfunctional and is s...	        102
#1	0	@user @user thanks for #lyft credit i can't us...	        122
#2	0	bihday your majesty	                                        21
#3	0	#model i love u take with u all the time in ...	                86
#4	0	factsguide: society now #motivation	                        39
#...	...	...	...
#31956	0	off fishing tomorrow @user carnt wait first ti...	        61
#31957	0	ate @user isz that youuu?Ã°??ÂÃ°??ÂÃ°??ÂÃ°??ÂÃ°??ÂÃ°...	68
#31958	0	to see nina turner on the airwaves trying to...	                131
#31959	0	listening to sad songs on a monday morning otw...	        63
#31961	0	thank you @user for you follow	                                32		    

#analyze negative sentiments
negative = tweets_df[tweets_df['label'] == 1]
		    
#print out negative tweets
negative


#label	tweet	length
#13	1	@user #cnn calls #michigan middle school 'buil...	74
#14	1	no comment! in #australia #opkillingbay #se...	101
#17	1	retweet if you agree!	22
#23	1	@user @user lumpy says i am a . prove it lumpy.	47
#34	1	it's unbelievable that in the 21st century we'...	104
...	...	...	...
#31934	1	lady banned from kentucky mall. @user #jcpenn...	59
#31946	1	@user omfg i'm offended! i'm a mailbox and i'...	82
#31947	1	@user @user you don't have the balls to hashta...	112
#31948	1	makes you ask yourself, who am i? then am i a...	87
#31960	1	@user #sikh #temple vandalised in in #calgary,...	67
		   

## PLOT THE WORDCLOUD - powerful tool for performing text visualizations on data
# convert sentences into a list
sentences = tweets_df['tweet'].tolist()

#print out list
sentences 
		    
#length of list
len(sentences)

#convert sentences to one string
setences_as_one_string = ' '.join(sentences)
		    
!pip install WordCloud #install WordCloud library
from wordcloud import WordCloud

#define the dimensions of the visual
plt.figure(figsize=(20,20))
#generate the WordCloud visual using the sentences_as_one_string as a parameter
plt.imshow(WordCloud().generate(sentences_as_one_string))
	   