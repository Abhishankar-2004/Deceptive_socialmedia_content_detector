import os
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
import joblib

# Load dataset
df = pd.read_excel(r'output_file.csv')  # Use raw string (r) to avoid unicode error

# Features and labels
X = df['Bio Keywords']
y = df['Legit/Deceptive']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build pipeline
model = make_pipeline(CountVectorizer(), MultinomialNB())

# Train model
model.fit(X_train, y_train)

# Save model
joblib.dump(model, 'deceptive_id_model.pkl')
