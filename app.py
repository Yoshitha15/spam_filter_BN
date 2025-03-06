import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score
import string

# Load dataset
file_path = 'mail_data.csv'
df = pd.read_csv(file_path)

# Hardcoded list of stopwords
stopwords_list = set("""a about above after again against all am an and any are aren't as at be because been before \
being below between both but by can't cannot could couldn't did didn't do does doesn't doing don't down during each few for from further \
had hadn't has hasn't have haven't having he he'd he'll he's her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into \
is isn't it it's its itself let's me more most mustn't my myself no nor not of off on once only or other ought our ours ourselves out over own same \
shan't she she'd she'll she's should shouldn't so some such than that that's the their theirs them themselves then there there's these they they'd they'll they're they've \
this those through to too under until up very was wasn't we we'd we'll we're we've were weren't what what's when when's where where's which while who who's \
whom why why's with won't would wouldn't you you'd you'll you're you've your yours yourself yourselves""".split())

# Preprocessing function
def transform_text_simple(text):
    text = text.lower()
    text = "".join(char for char in text if char.isalnum() or char.isspace())
    text = " ".join(word for word in text.split() if word not in stopwords_list)
    return text

# Apply preprocessing to the dataset
df['transformed_text'] = df['Message'].apply(transform_text_simple)

# Encode labels ('ham' = 0, 'spam' = 1)
df['Category'] = df['Category'].map({'ham': 0, 'spam': 1})

# Feature extraction using TF-IDF
tfidf = TfidfVectorizer(max_features=3000)
X = tfidf.fit_transform(df['transformed_text']).toarray()
y = df['Category'].values

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Multinomial Naive Bayes model
mnb = MultinomialNB()
mnb.fit(X_train, y_train)

# Evaluate the model
y_pred = mnb.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
precision = precision_score(y_test, y_pred)

# Print evaluation results
print("Accuracy:", accuracy)
print("Confusion Matrix:\n", conf_matrix)
print("Precision:", precision)

# Function for predicting if a message is spam or not
def predict_message(message):
    transformed_message = transform_text_simple(message)
    vectorized_message = tfidf.transform([transformed_message]).toarray()
    prediction = mnb.predict(vectorized_message)[0]
    return "Spam" if prediction == 1 else "Not Spam"

# Example: Take user input for classification
user_message = input("Enter a message to classify: ")
classification = predict_message(user_message)
print(f"The message is classified as: {classification}")
