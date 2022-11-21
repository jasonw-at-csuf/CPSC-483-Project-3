import pandas as pd
from nltk.corpus import stopwords
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import nltk
nltk.download('stopwords')


# import the dataset
df = pd.read_csv(
    "https://raw.githubusercontent.com/jasonw-at-csuf/CPSC-483-Project-3/main/emails.csv"
)

# data preprocessing
df.text = (
    df.text.str.replace("\W+", " ", regex=True)
    .str.replace("\s+", " ", regex=True)
    .str.strip()
)
df.text = df.text.str.lower()

stop = stopwords.words("english")

df.text.apply(lambda x: " ".join([word for word in x.split() if word not in (stop)]))
df

# split testing and training data

X_train, X_test, y_train, y_test = train_test_split(df.text.values, df.spam.values)


# extract features
vectorizer = CountVectorizer(ngram_range=(1, 2)).fit(X_train)
X_train_vectorized = vectorizer.transform(X_train)


nb_model = MultinomialNB(alpha=0.1)
nb_model.fit(X_train_vectorized, y_train)

svm_model = SVC(kernel="linear", random_state=0)
svm_model.fit(X_train_vectorized, y_train)

knn_model = KNeighborsClassifier(n_neighbors=15)
knn_model.fit(X_train_vectorized, y_train)

with open("messages.txt") as f:
    messages = f.readlines()
classifiers = {
    "Naive Bayes": nb_model,
    "SVM": svm_model,
    "KNN": knn_model,
}

results = pd.DataFrame.from_dict(
    {
        name: [
            "spam" if result else "not spam"
            for result in classifier.predict(vectorizer.transform(messages))
        ]
        for name, classifier in classifiers.items()
    }
)
results.insert(0, "message", [message[:40] for message in messages], True)
print(results)