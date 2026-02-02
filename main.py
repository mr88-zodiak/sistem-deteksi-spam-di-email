from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score
import pandas as pd
from nlp_id.lemmatizer import Lemmatizer
import time
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import nltk
nltk.download('punkt')



data = pd.read_csv('data_clean/dataset_gmail_cleaned.csv')


data = data.dropna(subset=['text','label'])


data['text'] = data['text'].astype(str).str.lower()


data['text'] = data['text'].apply(word_tokenize)


stopWords = ['a', 'adalah', 'adanya', 'adapun', 'agar', 'akan', 'akankah',
             'akhir', 'akhiri', 'akhirnya', 'aku', 'akulah', 'anda',
             'apa', 'apaan', 'apabila', 'apakah', 'atau', 'awal',
             'bagai', 'bagaimana', 'bagi', 'bahkan', 'bahwa',
             'baik', 'banyak', 'baru', 'bawah', 'beberapa']
start = time.time()

data = data.dropna(subset=['text', 'label'])


data['text'] = data['text'].astype(str).str.lower()


data['text'] = data['text'].apply(word_tokenize)


stopWords = ['a', 'adalah', 'adanya', 'adapun', 'agar', 'akan', 'akankah',
             'akhir', 'akhiri', 'akhirnya', 'aku', 'akulah', 'anda',
             'apa', 'apaan', 'apabila', 'apakah', 'atau', 'awal',
             'bagai', 'bagaimana', 'bagi', 'bahkan', 'bahwa',
             'baik', 'banyak', 'baru', 'bawah', 'beberapa']
start = time.time()


lemmatizer = Lemmatizer()
data['text'] = data['text'].apply(
    lambda x: [lemmatizer.lemmatize(word) for word in x]
)


data['text'] = data['text'].apply(lambda x: ' '.join(x))


X_train, X_test, y_train, y_test = train_test_split(
    data['text'],
    data['label'],
    test_size=0.2,
    random_state=42,
    stratify=data['label'] 
)


vectorizer = TfidfVectorizer(stop_words=stopWords, ngram_range=(1, 1))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)


model = MultinomialNB()
model.fit(X_train_tfidf, y_train)


y_pred = model.predict(X_test_tfidf)


label = ['ham', 'spam']
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label, yticklabels=label)
plt.show()
print(classification_report(y_test, y_pred))
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Time:", time.time()-start)