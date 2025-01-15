import pandas as pd
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer


# importing the Dataset
messages = pd.read_csv('smsspamcollection/SMSSpamCollection', sep='\t', names=["label", "message"])

print("printing the message: ",messages)
print(messages['message'].loc[451])


#Data cleaning and preprocessing
ps = PorterStemmer()


corpus = []
for i in range(0, len(messages)):
    review = re.sub('[^a-zA-Z]', ' ', messages['message'][i])
    review = review.lower()
    review = review.split()
    
    review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)
    
    
print("print the corpus: ", corpus)
print()

cv = CountVectorizer(max_features=2500)
X = cv.fit_transform(corpus).toarray()

y=pd.get_dummies(messages['label'])
y=y.iloc[:,1].values


# Train Test Split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)


from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(X_train, y_train)


y_pred=spam_detect_model.predict(X_test)


from sklearn.metrics import accuracy_score,classification_report

score=accuracy_score(y_test,y_pred)
print("using Bag of words model: ")
print()
print("Score: ",score)
print(classification_report(y_pred,y_test))
print()


# Creating the TFIDF model
tv = TfidfVectorizer(max_features=2500)
X = tv.fit_transform(corpus).toarray()


# Train Test Split

X_train1, X_test1, y_train1, y_test1 = train_test_split(X, y, test_size = 0.20, random_state = 0)
spam_detect_model = MultinomialNB().fit(X_train, y_train)


#prediction
y_pred1=spam_detect_model.predict(X_test1)

print("using TFIDF model: ")
print()
score1=accuracy_score(y_test1,y_pred1)
print("Score: ",score1)
print(classification_report(y_pred1,y_test1))