import numpy as np
import pandas as pd
import nltk
import string
import pickle
import warnings
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

#------------------------------------------------------------
# PART 1: SETUP AND DATA LOADING
#------------------------------------------------------------

# Download necessary NLTK resources
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading NLTK resources...")
    nltk.download('punkt')
    nltk.download('stopwords')
    nltk.download('wordnet')

print("Loading and preparing data...")

# Load dataset
try:
    df = pd.read_csv('spam.csv', encoding='latin-1')
    print(f"Dataset loaded with {len(df)} rows")
except Exception as e:
    print(f"Error loading dataset: {e}")
    print("Trying with different encoding...")
    try:
        df = pd.read_csv('spam.csv', encoding='utf-8')
        print(f"Dataset loaded with {len(df)} rows")
    except Exception as e:
        print(f"Error loading dataset with utf-8 encoding: {e}")
        raise

#------------------------------------------------------------
# PART 2: DATA PREPROCESSING
#------------------------------------------------------------

# Clean up data
df = df.drop(columns=['Unnamed: 2', 'Unnamed: 3', 'Unnamed: 4'], errors='ignore')
df.rename(columns={'v1': 'target', 'v2': 'text'}, inplace=True)

# Encode target
encoder = LabelEncoder()
df['target'] = encoder.fit_transform(df['target'])

# Drop duplicates
df = df.drop_duplicates(keep='first')

# Initialize stop words
stop_words = set(stopwords.words('english'))

# Advanced text preprocessing with lemmatization
lemmatizer = WordNetLemmatizer()

def advanced_preprocess(text):
    """
    Perform advanced text preprocessing including:
    - Lowercasing
    - Tokenization
    - Removing special characters and numbers
    - Removing stopwords and punctuation
    - Lemmatization (more accurate than stemming)
    """
    try:
        # Handle NaN values
        if pd.isna(text):
            return ""
        
        # Convert to lowercase and tokenize
        tokens = word_tokenize(str(text).lower())
        
        # Keep only alphabetic tokens
        tokens = [word for word in tokens if word.isalpha()]
        
        # Remove stopwords and punctuation
        tokens = [word for word in tokens if word not in stop_words and word not in string.punctuation]
        
        # Lemmatization (better than stemming)
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        
        return " ".join(tokens)
    except Exception as e:
        print(f"Error preprocessing text: {e}")
        return ""

print("Applying advanced preprocessing...")
df['transformed_text'] = df['text'].apply(advanced_preprocess)

#------------------------------------------------------------
# PART 3: FEATURE EXTRACTION
#------------------------------------------------------------

# Feature extraction with optimized TF-IDF
print("Vectorizing text with TF-IDF...")
tfidf = TfidfVectorizer(
    max_features=3000,      # Number of features to keep
    ngram_range=(1, 2),     # Include unigrams and bigrams
    min_df=2,               # Minimum document frequency
    max_df=0.9,             # Maximum document frequency
    sublinear_tf=True       # Apply sublinear tf scaling
)

X = tfidf.fit_transform(df['transformed_text'])
y = df['target'].values

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

#------------------------------------------------------------
# PART 4: MODEL TRAINING AND EVALUATION
#------------------------------------------------------------

print("Training multiple models...")

# Optimized Logistic Regression
lr = LogisticRegression(
    C=10,                   # Regularization strength
    solver='liblinear',     # Algorithm
    penalty='l1',           # L1 regularization for feature selection
    max_iter=1000,          # Maximum iterations
    class_weight='balanced' # Handle class imbalance
)

# MultinomialNB - Very efficient for text classification
nb = MultinomialNB(alpha=0.1) # Smoothing parameter

# Random Forest with tuned parameters
rf = RandomForestClassifier(
    n_estimators=50,        # Number of trees
    max_depth=10,           # Tree depth
    min_samples_split=5,    # Min samples to split a node
    min_samples_leaf=2,     # Min samples at leaf node
    max_features='sqrt',    # Features to consider for best split
    class_weight='balanced', # Handle class imbalance
    random_state=42,
    n_jobs=-1               # Use all processors
)

# Extra Trees Classifier
etc = ExtraTreesClassifier(
    n_estimators=50,        # Number of trees
    max_depth=10,           # Tree depth
    min_samples_split=5,    # Min samples to split
    class_weight='balanced', # Handle class imbalance
    random_state=42,
    n_jobs=-1               # Use all processors
)

# Define models dictionary
models = {
    'Logistic Regression': lr,
    'Multinomial NB': nb,
    'Random Forest': rf,
    'Extra Trees': etc
}

# Train and evaluate each model
results = []
for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # Store results
    results.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-score': f1
    })
    
    print(f"{name} - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

#------------------------------------------------------------
# PART 5: ENSEMBLE MODEL
#------------------------------------------------------------

# Create a Voting Classifier (ensemble)
voting_clf = VotingClassifier(estimators=[
    ('lr', models['Logistic Regression']),
    ('nb', models['Multinomial NB']),
    ('rf', models['Random Forest']),
    ('et', models['Extra Trees'])
], voting='soft')

voting_clf.fit(X_train, y_train)
y_pred = voting_clf.predict(X_test)

# Evaluate ensemble
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

results.append({
    'Model': 'Voting Ensemble',
    'Accuracy': accuracy,
    'Precision': precision,
    'Recall': recall,
    'F1-score': f1
})

print(f"Voting Ensemble - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}")

#------------------------------------------------------------
# PART 6: RESULTS SUMMARY AND MODEL SELECTION
#------------------------------------------------------------

# Create results dataframe and display
results_df = pd.DataFrame(results)
print("\nModel Comparison:")
print(results_df)

# Determine best model based on F1 score
best_model_idx = results_df['F1-score'].idxmax()
best_model_name = results_df.loc[best_model_idx, 'Model']
print(f"\nBest model based on F1-score: {best_model_name}")

# Save the best model and vectorizer
if best_model_name == 'Voting Ensemble':
    best_model = voting_clf
else:
    best_model = models[best_model_name]

print("Saving best model and vectorizer...")
pickle.dump(tfidf, open('vectorizer.pkl', 'wb'))
pickle.dump(best_model, open('model.pkl', 'wb'))

print("Model training and selection complete. The best model has been saved.")

#------------------------------------------------------------
# PART 7: VISUALIZATION
#------------------------------------------------------------

# Generate confusion matrix for the best model
try:
    plt.figure(figsize=(8, 6))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title(f'Confusion Matrix - {best_model_name}')
    plt.savefig('confusion_matrix.png')
    print("Confusion matrix saved as 'confusion_matrix.png'")
except Exception as e:
    print(f"Error generating confusion matrix: {e}")