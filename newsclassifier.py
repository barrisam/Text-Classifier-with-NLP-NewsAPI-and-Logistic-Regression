from newsapi import NewApiClient
import pandas as pd
import re
import nltk
nltk.download('punkt')
from nltk.tokenize import word_tokenize
nltk.download('stopwords')
from nltk.corpus import stopwords
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score


#to get api key
newsapi = NewsApiClient(api_key='2a7dd9f4dd8fxxxxxxxxxxxxxxxxx') # Get your API key from NewsAPI

#using the key to get tech articles
tech_articles = newsapi.get_everything(q='tech', language='en', page_size=100)
print(tech_articles)

#checking the keys in the output
print(tech_articles.keys())
#dict_keys(['status', 'totalResults', 'articles'])

#transform to a pandas dataframe
tech = pd.DataFrame(tech_articles['articles'])
print(tech)

#add category to the dataframe
tech['category'] = 'Tech'
print(tech)

#add more categories
entertainment_articles = newsapi.get_everything(q='entertainment',language='en', page_size=100)
business_articles = newsapi.get_everything(q='business',language='en', page_size=100)
sports_articles = newsapi.get_everything(q='sports',language='en', page_size=100)
politics_articles = newsapi.get_everything(q='politics',language='en', page_size=100)
travel_articles = newsapi.get_everything(q='travel',language='en', page_size=100)
food_articles = newsapi.get_everything(q='food',language='en', page_size=100)
health_articles = newsapi.get_everything(q='health',language='en', page_size=100)


#transform them to a dataframe
entertainment = pd.DataFrame(entertainment_articles['articles'])
entertainment['category'] = 'Entertainment'
business = pd.DataFrame(business_articles['articles'])
business['category'] = 'Business'
sports = pd.DataFrame(sports_articles['articles'])
sports['category'] = 'Sports'
politics = pd.DataFrame(politics_articles['articles'])
politics['category'] = 'Politics'
travel = pd.DataFrame(travel_articles['articles'])
travel['category'] = 'Travel'
food = pd.DataFrame(food_articles['articles'])
food['category'] = 'Food'
health = pd.DataFrame(health_articles['articles'])
health['category'] = 'Health

# merge everything into one dataframe
categories = [tech, entertainment, business, sports, politics, travel, food, health]
df = pd.concat(categories)
print(df.info())



# Define the function to clean the news title column
def cleaned_desc_column(text):
    # Remove commas
    text = re.sub(r',', '', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Remove full stops
    text = re.sub(r'\.', '', text)
    # Remove single quotes and double quotes
    text = re.sub(r"['\"]", '', text)
    # Remove other non-word characters
    text = re.sub(r'\W', ' ', text)

    text_token = word_tokenize(text) 
    stop_words = set(stopwords.words('english'))

    filtered_text = [] 

    for sw in text_token:
      if sw not in stop_words:
          filtered_text.append(sw)

    text = " ".join(filtered_text)
    return text
  
# Apply the clean_text_column function to the text_column in the DataFrame
df['news_title'] = df['title'].apply(cleaned_desc_column)
print(df)
# The cleaned column 'news_title' is added to the dataframe. 

#getting the category we need for testing
X = df['news_title']
y = df['category']

#spliting data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 90)
print(X_train.shape)
print(X_test.shape)

#creating a pipeline to build the classifier
lr = Pipeline([('vect', CountVectorizer()),
               ('tfidf', TfidfTransformer()),
               ('clf', LogisticRegression(max_iter = 1000)),
              ])

# Train the logistic regression model on the training set
lr.fit(X_train,y_train)

# Make predictions on the test set
y_pred = lr.predict(X_test)

# Calculate the accuracy of the model
print(f"Accuracy is : {accuracy_score(y_pred,y_test)}")

#Output
#Accuracy is : 0.7208333333333333

# Test The Model With Different Articles
news = ["Biden to Sign Executive Order That Aims to Make Child Care Cheaper",
       "Google Stock Loses $57 Billion Amid Microsoft’s AI ‘Lead’—And Reports It Could Be Replaced By Bing On Some Smartphones",
       "Poland suspends food imports from Ukraine to assist its farmers",
       "Can AI Solve The Air Traffic Control Problem? Let's Find Out",
       "Woman From Odisha Runs 42.5 KM In UK Marathon Wearing A Saree",
       "Hillary Clinton: Trump cannot win election - but Biden will",
       "Jennifer Aniston and Adam Sandler starrer movie 'Murder Mystery 2' got released on March 24, this year"]

predicted = lr.predict(news)

for doc, category in zip(news, predicted):
     print(category)
"""
Health
Tech
Food
Tech
Sports
Politics
Entertainment
"""
