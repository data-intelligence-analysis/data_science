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
	   
		    
## Plot the wordcloud of the "negative" dataframe
# Print negative tweets		   
negative
# convert negative sentences to a list 
sentences_negative = negative['tweet'].tolist()
# print negative sentences in a list form
sentences_negative
# join stences as one string
negative_sentences_as_one_string = ' '.join(sentences_negative)
# print negative sentences as a string
negative_sentences_as_one_string
# word cloud visual of the negative sentences / sentiments
plt.figure(figsize=(20,30))
plt.imshow(WordCloud().generate(negative_sentences_as_one_string))
		    
##What do you notice? Does the data make sense?
##Yes. There are more negative words like hate, racist etc.
		    
#PERFORM DATA CLEANING - REMOVE DATA PUNCTUATION FROM TEXT

import string
string.punctuation
#test striing
Test = 'Good morning beautiful people :)... I am having fun learning Machine learning and AI!!'
#Remove Puntuations from string using list comprehension
test_punc_removed = [ char for char in Test if char not in string.punctuation ]

test_punc_removed 
['G',
 'o',
 'o',
 'd',
 ' ',
 'm',
 'o',
 'r',
 'n',
 'i',
 'n',
 'g',
 ' ',
 'b',
 'e',
 'a',
 'u',
 't',
 'i',
 'f',
 'u',
 'l',
 ' ',
 'p',
 'e',
 'o',
 'p',
 'l',
 'e',
 ' ',
 ' ',
 'I',
 ' ',
 'a',
 'm',
 ' ',
 'h',
 'a',
 'v',
 'i',
 'n',
 'g',
 ' ',
 'f',
 'u',
 'n',
 ' ',
 'l',
 'e',
 'a',
 'r',
 'n',
 'i',
 'n',
 'g',
 ' ',
 'M',
 'a',
 'c',
 'h',
 'i',
 'n',
 'e',
 ' ',
 'l',
 'e',
 'a',
 'r',
 'n',
 'i',
 'n',
 'g',
 ' ',
 'a',
 'n',
 'd',
  ' ',
 'A',
 'I']

# Use join to create the string
test_punc_removed_joined = ''.join(test_punc_removed)
test_punc_removed_joined
		    
#Remove punctuations using a different method
import string
punc = string.punctuation
string_list = []
for char in Test:
    if char not in punc:
        string_list.append(char)
        string_join = ''.join(string_list)
print (string_join)
		    
#Answer: G o o d   m o r n i n g   b e a u t i f u l   p e o p l e     I   a m   h a v i n g   f u n   l e a r n i n g   M a c h i n e   l e a r n i n g   a n d   A I
'''
import string
punc = string.punctuation
string_list = []
for char in Test:
    if char not in punc:
        string_list.append(char)

string_join = ''.join(string_list)
string_join''' - #alternative solution
# Answer: 'G o o d   m o r n i n g   b e a u t i f u l   p e o p l e     I   a m   h a v i n g   f u n   l e a r n i n g   M a c h i n e   l e a r n i n g   a n d   A I'
		    
## PERFORM DATA CLEANING - REMOVE STOPWORDS
import nltk # Natural Language tool kit

nltk.download('stopwords')

#you have to donwload stopwords Package to execute this command		    
from nltk.corpus import stopwords
stopwords.words('english') - # https://www.geeksforgeeks.org/removing-stop-words-nltk-python/

#List comprehension 
test_punc_removed_join_clean = [word for word in test_punc_removed_join.split() if word.lower() not in stopwords.words('english')

test_punc_removed_join_clean
			

#For the following text, create a pipeline to remove punctuations followed by removing stopwords

mini_challenge = "Here is a mini challenge, that will teach you how to remove stopwords and puctuations"
#List comprehension
mini_challenge_remove_punc = [char for char in mini_challenge if char not in string.punctuations]
#Join elements of list as string 
mini_challenge_remove_punc_join = ''.join(mini_challenge_remove_punc)
#remove stopwords and punctuations
mini_challenge_remove_punc_join_clean = [word for word in mini_challenge_remove_punc_join if word.lower not in stopwords.words('english')]
mini_challenge_remove_punc_join_clean
				
# output = ['mini', 'challenge', 'teach', 'remove', 'stopwords', 'punctuations']

# PERFORM COUNT VECTORIZATION (TOKENIZATION)
from sklearn.feature_extraction.text import CountVectorizer
sample_data = ['This is the first paper.','This paper is the second paper.','And this is the third one.','Is this the first paper?']

#count vectorizer function
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(sample_data)

# get the unique names in the data				
print(vectorizer.get_feature_names())
#output = ['and','first','is','one','paper','second','the','third','this']
				
print(X.toarray())
'''
[[0 1 1 0 1 0 1 0 1]
 [0 0 1 0 2 1 1 0 1]
 [1 0 1 1 0 0 1 1 1]
 [0 1 1 0 1 0 1 0 1]]
 '''
mini_challenge = ['Hello World',
		  'Hello Hello World',
		  'Hello World world world']
#y= vectorizer.fit_transform(mini_challenge)
'''output = [[1,1]
	    [2,1]
	    [1,3]]'''



y = vectorizer.fit_transform(mini_challenge)
print(vectorizer.get_feature_names)
#output = ['hello', 'world']
print(y.toarray())
'''output = [[1,1]
	    [2,1]
	    [1,3]]'''
				
## CREATE A PIPELINE TO REMOVE PUNCTUATIONS, STOPWORDS AND PERFORM COUNT VECTORIZATION
#Lets define a function to remove and clean up all the messages
# The function removes punctuations and stopwords
def message_cleaning(message):
    test_punc_removed = [ char for char in message if char not in string.punctuation ]
    test_punc_removed_join = ''.join(test_punc_removed)
    test_punc_removed_join_split = test_punc_removed_join.split()
    test_punc_removed_join_clean = [word for word in test_punc_removed_join_split if word.lower() not in stopwords.words('english')
    return test_punc_removed_join_clean
				    
#Aplly the message cleaning function to the existing tweets in the file
tweets_clean_df = tweets_df['tweet'].apply(message_cleaning)
#show the cleaned up version at index 5
print(tweet_clean_df[5])
				    
#show the original version
print(tweet_df["tweet"][5])

from sklearn.feature_extraction.test import CountVectorizer
#define the cleaning process 
vectorizer = CountVectorizer(analyzer = message_cleaning)
tweets_countervectorizer = CountVectorizer(analyze = message_cleaning, dtype = 'unit8').fit_transformer(tweets_df['tweet']).toarray()
tweets_countervectorizer.shape()
#(31962, 47386)
X = tweets_countervectorizer
y = tweets_df['label']
				    
				    
## UNDERSTAND THE CONCEPTS BEHIND THE THEORY AND INTUITION BEHIND NAIVE BAYES CLASSIFIER

#Mini Challenge - Naive Bayes: Quiz/Calculate the probability of non-retiring (RED CLASS)
				    
#P(No Retire|x) = (P(X|No Retire) * P(No Retire))/P(X)  = 1 - P(Retiring|x) = 1 - 0.25 = 0.75
# P(X|No Retire) (# of No Retiring/Total Points) = 3/20
# P(No Retire) (# of similar observations for no retiring / Total # no retiring) = 20/60 = 1/3
# P(X) (# of similar observations / total # points = 4/60
# P(No Retire|x) = (3/20 * 1/3)/1/15 = 15/20 = 0.75
				    
#Task #10: TRAIN A NAIVE BAYES CLASSIFIER MODEL
#tokenization of all the tweets
X.shape
#(31962, 47386)
#All the sentiments of zeroes (0) and one's (1)
y.shape
#(31962,)
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2)

from sklearn.naive_bayes import MultinomialNB
NB_classifier = MultinomialNB()
NB_classifier.fit(X_train, y_train)
#MultinomialNB(alpha=1.0, class_prior=None, fit_prior=True)

#Assess Trained Model Performance
from sklearn.matrics import classification_report, confusion_matrix
#Predicting the Test set results
y_predict_test = NM_classifier.predict(X_test)
cm = confusion_matrix(y_test, y_predict_test)
sns.heatmap(cm, annot = True)
#<matplotlib.axes._subplots.AxesSubplot at 0x1dc38033288>

print(classification_report(y_test, y_predict_test)
      

#Mini challenge - retrain and reassess the model using a different type of classifier such as a logistic regression classifier
from sklearn.linear_model import LogisticRegression #https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html
log_reg = LogisticRegression(random_state=0).fit(X_train, y_train)
from sklearn.matrics import classification_report, confusion_matrix
#Predicting the Test set results
y_predict_log_test = log_reg.predict(X_test)
cm = confusion_matrix(y_test, y_predict_log_test)
sns.heatmap(cm, annot = True)
