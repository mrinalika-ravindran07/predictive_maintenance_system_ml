import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# 1. CREATE DUMMY DATASET
data = {
    'text': [
        "Free money now!!!", 
        "Hi Bob, how about lunch?", 
        "Win a generic lottery prize", 
        "Meeting schedule for tomorrow", 
        "Cheap meds and pills", 
        "Project deadline is Friday", 
        "Click here for a free vacation", 
        "Can we reschedule the call?"
    ],
    'label': [
        1, # Spam
        0, # Not Spam
        1, 
        0, 
        1, 
        0, 
        1, 
        0
    ]
}
df = pd.DataFrame(data)

print("--- Sample Data ---")
print(df.head())

# 2. PREPROCESSING: Turn Text into Numbers (Vectorization)
# We use CountVectorizer to create a "Bag of Words"
vectorizer = CountVectorizer()
X = vectorizer.fit_transform(df['text'])
y = df['label']

print(f"\nVocabulary Size: {len(vectorizer.get_feature_names_out())} words")
print(f"Feature Names (first 5): {vectorizer.get_feature_names_out()[:5]}")

# 3. TRAIN MODEL
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

clf = MultinomialNB() # Naive Bayes is standard for text counts
clf.fit(X_train, y_train)

# 4. TEST WITH NEW EMAILS
new_emails = [
    "Hey, are you free for dinner?",      # Should be Not Spam (0)
    "WIN FREE MONEY NOW click here"       # Should be Spam (1)
]
new_emails_vectorized = vectorizer.transform(new_emails)
predictions = clf.predict(new_emails_vectorized)

print("\n--- Predictions on New Emails ---")
for text, label in zip(new_emails, predictions):
    result = "SPAM" if label == 1 else "Not Spam"
    print(f"Email: '{text}' -> Prediction: {result}")