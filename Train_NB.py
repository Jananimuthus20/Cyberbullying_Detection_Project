import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics
from bs4 import BeautifulSoup
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from imblearn.over_sampling import SMOTE

df = pd.read_csv('train.csv')

def remove_html_tags(text):
    soup = BeautifulSoup(text, 'html.parser')
    return soup.get_text()

df['comment_text'] = df['comment_text'].apply(remove_html_tags)

X_train, X_test, y_train, y_test = train_test_split(df['comment_text'], df['toxic'], test_size=0.2, random_state=42)

vectorizer = TfidfVectorizer(max_df=0.05, min_df=1, ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train_tfidf, y_train)

model = MultinomialNB()
model.fit(X_train_res, y_train_res)

# Make predictions on the test set
y_pred = model.predict(X_test_tfidf)

accuracy = metrics.accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy:.2f}")

# Classification Report
classification_report_result = metrics.classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_result)

# Confusion Matrix
confusion_matrix_result = metrics.confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion_matrix_result)

plt.figure(figsize=(8, 6))
sns.heatmap(confusion_matrix_result, annot=True, fmt='d', cmap='Blues', xticklabels=['Not Toxic', 'Toxic'], yticklabels=['Not Toxic', 'Toxic'])
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()

y_proba = model.predict_proba(X_test_tfidf)[:, 1]
fpr, tpr, _ = metrics.roc_curve(y_test, y_proba)
roc_auc = metrics.auc(fpr, tpr)

plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend(loc='lower right')
plt.show()

joblib.dump(model, 'naive_bayes_model.pkl')
joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')  # Save the vectorizer
