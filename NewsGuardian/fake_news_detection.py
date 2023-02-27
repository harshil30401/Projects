import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
# from sklearn.metrics import classification_report
import re, string


true = pd.read_csv("true.csv")
fake = pd.read_csv("fake.csv")

fake["class"] = 0
true["class"] = 1

fake.head()

true.head()

fake.shape, true.shape

fake_manual_testing = fake.tail(10)
for i in range(23480, 23470, -1):
    fake.drop([i], axis=0, inplace=True)
true_manual_testing = true.tail(10)
for i in range(21416, 21406, -1):
    true.drop([i], axis=0, inplace=True)

manual_testing = pd.concat([fake_manual_testing, true_manual_testing], axis=0)
manual_testing.to_csv("News.csv")

news = pd.concat([fake, true], axis = 0)

news.drop(['title', 'subject', 'date'], axis=1, inplace=True)
news

#shuffle dataframe
news = news.sample(frac=1)
news

news.isnull().sum()

#removing special characters

def word_drop(text):
  text = text.lower()
  text = re.sub('\[.*?\]','', text)
  text = re.sub("\\W", " ", text)
  text = re.sub('https?://\S+|www\.\S+', '', text)
  text = re.sub('<.*?>+', '', text)
  text = re.sub('[%s]' % re.escape(string.punctuation), '', text)
  text = re.sub('\w*\d\w*', '', text)
  return text

news["text"] = news["text"].apply(word_drop)

news

x = news["text"]
y = news["class"]

# splitting into test and train

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = .25)

from sklearn.feature_extraction.text import TfidfVectorizer

vectorization = TfidfVectorizer()
xv_train = vectorization.fit_transform(x_train)
xv_test = vectorization.transform(x_test)

"""### Logistic Regression"""

from sklearn.linear_model import LogisticRegression

logisticRegression = LogisticRegression()
logisticRegression.fit(xv_train, y_train)

logisticRegression.score(xv_test, y_test)

pred_logisticRegression = logisticRegression.predict(xv_test)

# print(classification_report(y_test, pred_logisticRegression))

"""### Decision Tree Classifier"""

from sklearn.tree import DecisionTreeClassifier

decisionTree = DecisionTreeClassifier()
decisionTree.fit(xv_train, y_train)

decisionTree.score(xv_test, y_test)

pred_decisionTree = decisionTree.predict(xv_test)

# print(classification_report(y_test, pred_decisionTree))

"""### Gradient Boosting Classifier"""

from sklearn.ensemble import GradientBoostingClassifier

gradientBoostingClassifier = GradientBoostingClassifier(random_state=0)
gradientBoostingClassifier.fit(xv_train, y_train)

gradientBoostingClassifier.score(xv_test, y_test)

pred_gradientBoostingClassifier = gradientBoostingClassifier.predict(xv_test)
# print(classification_report(y_test, pred_gradientBoostingClassifier))

"""### Random Forest Classifier"""

from sklearn.ensemble import RandomForestClassifier

randomForestClassifier = RandomForestClassifier(random_state=0)
randomForestClassifier.fit(xv_train, y_train)

randomForestClassifier.score(xv_test, y_test)

pred_randomForestClassifier=  randomForestClassifier.predict(xv_test)

# print(classification_report(y_test, pred_randomForestClassifier))

"""### Manual Testing"""

def output(n):
  if n == 0:
    return "Fake"
  elif n == 1:
    return "True"


def manual_testing(news):
  testing_news = {"text": [news]}
  new_def_test = pd.DataFrame(testing_news)
  new_def_test["text"] = new_def_test["text"].apply(word_drop)
  new_x_test = new_def_test["text"]
  new_xv_test = vectorization.transform(new_x_test)
  pred_logisticRegression = logisticRegression.predict(new_xv_test)
  pred_decisionTree = decisionTree.predict(new_xv_test)
  pred_gradientBoostingClassifier = gradientBoostingClassifier.predict(new_xv_test)
  pred_randomForestClassifier = randomForestClassifier.predict(new_xv_test)

  # return print(
  #     "\n\nLogistic Regression: {} \nDecision Tree Classifier: {} \nGradient Boosting Classifier: {} \nRandom Forest Classifier: {}".format(
  #       output(pred_logisticRegression),
  #       output(pred_decisionTree),
  #       output(pred_gradientBoostingClassifier),
  #       output(pred_randomForestClassifier)  
  #     ))

  return print(output(pred_decisionTree))


print("Hey there i am KITT, Beware of fake news and don't send it across!")  
headline = str(input("Tell me a news: "))
manual_testing(headline)

