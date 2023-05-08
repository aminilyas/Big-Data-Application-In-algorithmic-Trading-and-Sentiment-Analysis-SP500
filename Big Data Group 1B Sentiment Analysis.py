# -*- coding: utf-8 -*-

"""
#########################################################################################################
Created on Wed Feb  8 21:59:44 2023


PREPARED FOR:
    BIG DATA FOR FINANCE COURSE

PREPARED BY: 
    GROUP 1 
    Amin ILYAS, Nizar AQACHMAR, Pritam RITU RAJ, Zahi SAMAHA

PROJECT TITLE:
    A DEMONSTRATION ON BIG DATA USAGE AND ITS APPLICATIONS IN ALGORITHMIC TRADING AND SENTIMENT ANALYSIS: 
    A CASE STUDY OF S&P 500 STOCKS
    
CODE PART 2:
    SENTIMENT ANALYSIS 

#########################################################################################################
"""

#%%
# Import libraries
import snscrape.modules.twitter as sntweeter
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import matplotlib.pyplot as plt


# load the model and tokenizer
roberta = "cardiffnlp/twitter-roberta-base-sentiment"
model = AutoModelForSequenceClassification.from_pretrained(roberta)
tokenizer = AutoTokenizer.from_pretrained(roberta)
labels = ['Negative', 'Neutral', 'Positive']


# Set up a keyword and period 
query = "TSLA until:2023-01-31 since:2023-01-29 lang:en"
tweets = []
limit = 1000

# Twitter scrapping
for tweet in sntweeter.TwitterSearchScraper(query).get_items():
  if len(tweets) == limit:
    break
  else:
    tweets.append([tweet.date, tweet.username, tweet.content, tweet.likeCount])

# Create a dataframe for teh tweets
df = pd.DataFrame(tweets, columns=['Date', 'User', 'Tweet', 'likeCount'])
df2 = df.iloc[:,2]

# Sentiment analysis
l2 = []
for i in df2:
    encoded_tweet = tokenizer(i, return_tensors='pt')
    output = model(encoded_tweet['input_ids'], encoded_tweet['attention_mask'])

    scores = output[0][0].detach().numpy()
    scores = softmax(scores)

    for i in range(len(scores)):
    
        l = labels[i]
        s = scores[i]
        print(l,s)
        l2.append(l)
        l2.append(s)

neg = []
neu = []
pos = []

n = len(l2)
for i in range(0, len(l2)-1):
    if l2[i] == 'Negative':
        neg.append(l2[i+1])
    elif l2[i] == 'Neutral':
        neu.append(l2[i+1])
    elif l2[i] == 'Positive':
        pos.append(l2[i+1])

neg = pd.DataFrame(neg)
neu = pd.DataFrame(neu)
pos = pd.DataFrame(pos)

neg_des = neg.describe()
neu_des = neu.describe()
pos_des = pos.describe()

# Neutral words
x = round(neu_des.iloc[1,0]*100/2,2)

n = round(neg_des.iloc[1,0]*100+x,2)
p = round(pos_des.iloc[1,0]*100+x,2)

print("\n")
print("Negative:", str(n)+'%')
print("Positive:", str(p)+'%')


# Plot chart
p1=p
n1=n
# sample data
labels = [str(n1)+'%',str(p1)+'%']
sizes = [n1,p1]
colors = ['#ef5350', '#66bb6a']

# create half pie chart
fig, ax = plt.subplots()
fig.set_size_inches(8, 6)  # set the size to 8 inches by 6 inches
ax.axis('equal')
ax.set_title('Sentiment Analysis')
ax.pie(sizes, colors=colors, startangle=90, counterclock=False, wedgeprops={'width': 0.5})
plt.legend(labels)

# add circle in the center to create a half pie chart
center_circle = plt.Circle((0,0),0.70,fc='white')
fig.gca().add_artist(center_circle)

# rotate the plot by 180 degrees
plt.gca().invert_yaxis()
plt.gca().invert_xaxis()

# Display the plot
plt.show()