import pickle
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report

# Load the dataset
data = pd.read_csv('/home/user/shaktimaangpt/ml_engine/awesomechatgpt_prompts.csv')

# Check for data imbalance
print(data['act'].value_counts())

# Separate majority and minority classes
df_majority = data[data.act == "Life Coach"]
df_minority = data[data.act != "Life Coach"]  # Example for a minority class

# Upsample minority class (if needed)
df_minority_upsampled = resample(df_minority, 
                                 replace=True,    # sample with replacement
                                 n_samples=len(df_majority), # match majority class
                                 random_state=42) 

# Combine majority class with upsampled minority class
data_balanced = pd.concat([df_majority, df_minority_upsampled])

# Split the data into features and target
X = data_balanced['prompt']  # Features
y = data_balanced['act']     # Target

# Convert text data into numerical data using TfidfVectorizer
vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Set up Grid Search for RandomForestClassifier
param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}

model = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(model, param_grid, cv=5, scoring='f1_weighted')
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

# Evaluate the model
y_pred = best_model.predict(X_test)
print(classification_report(y_test, y_pred))

# Save the model and the vectorizer to pickle files
model_filename = '/home/user/shaktimaangpt/ml_engine/model.pkl'
vectorizer_filename = '/home/user/shaktimaangpt/ml_engine/vectorizer.pkl'
pickle.dump(best_model, open(model_filename, 'wb'))
pickle.dump(vectorizer, open(vectorizer_filename, 'wb'))

# Function to make predictions
def predict_act(prompt):
    # Load the vectorizer and model
    vectorizer = pickle.load(open(vectorizer_filename, 'rb'))
    model = pickle.load(open(model_filename, 'rb'))

    # Transform the input using the vectorizer (wrap it in a list)
    X_new = vectorizer.transform([prompt])

    # Make a prediction
    prediction = model.predict(X_new)
    return prediction[0]

# Example usage
if __name__ == "__main__":
    new_prompt = input("User Prompt: ")
    predicted_act = predict_act(new_prompt)
    print(f"Predicted act: {predicted_act}")
