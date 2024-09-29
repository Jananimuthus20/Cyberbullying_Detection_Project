import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn import metrics
from bs4 import BeautifulSoup
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

# Load the dataset
df = pd.read_csv('train.csv')

# Function to remove HTML tags from the 'comment_text' column
def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

# Apply the function to the 'comment_text' column in your dataset
df['comment_text'] = df['comment_text'].apply(remove_html_tags)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['toxic'], test_size=0.2, random_state=42)

# Initialize the TfidfVectorizer and transform the training and test data
vectorizer = TfidfVectorizer(max_df=0.05, min_df=1, ngram_range=(1, 2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Handle class imbalance with SMOTE
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

# Initialize the LinearSVC model
model = LinearSVC()

# Train the LinearSVC model
model.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

# Evaluate the performance of the model
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification Report
classification_report_result = metrics.classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_result)

# Confusion Matrix
confusion_matrix_result = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_matrix_result)

# Plot the Confusion Matrix
plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Toxic', 'Toxic'], yticklabels=['Not Toxic', 'Toxic'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# Save the trained model and the vectorizer to files
joblib.dump(model, 'linear_svc_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # Save the vectorizer
