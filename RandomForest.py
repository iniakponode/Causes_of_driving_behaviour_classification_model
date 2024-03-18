# Optimized version of the user's code for handling the dataset and training a Random Forest classifier

import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import cross_val_score

# Load data
df = pd.read_csv('F_Unc_PhD_no_data_stars_removed_dataset.csv')  # Update the dataset path

# Remove the specified columns
df = df.drop(['SpeedLimit', 'driver_id', 'Address', 'track_id', 'yaw', 'lateral', 'pointDate', 'tickTimestamp'], axis=1)

# Second Feature Engineering step, this is to make the model location and distance travelled agnostic.
df = df.drop(['longitude', 'latitude', 'totalMeters'], axis=1)

# Encode categorical columns - Assuming 'driver_type' is the only categorical column
df = pd.get_dummies(df, columns=['driver_type'])

# Label encoding for the target variable if it's categorical
le = LabelEncoder()
df['influencer'] = le.fit_transform(df['influencer'])

# Split the dataset into features and target variable
X = df.drop('influencer', axis=1)
y = df['influencer']
feature_names = X.columns
# Standardizing the features - important for Random Forest
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Third feature feature engineering step using SMOTE
# Apply SMOTE
smote = SMOTE(random_state=95)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)

rf_classifier = RandomForestClassifier(
    n_estimators=200,  # Try 50, 150, etc.
    max_depth=15,  # Experiment with different depths like 5, 15, None
    min_samples_split=10,  # Try different values like 2, 6, 10
    min_samples_leaf=3,  # Experiment with 1, 3, 5
    max_features='sqrt',  # Options: 'auto', 'sqrt', 'log2', or a fraction of total features
    random_state=42,
    # class_weight='balanced'
)


# # Initialize the Random Forest Classifier
# rf_classifier = RandomForestClassifier(random_state=42, n_estimators=100, max_depth=None)  # Default settings for RandomForest

# First Adjusted hyperparameters
# rf_classifier = RandomForestClassifier(
#     n_estimators=100,  # Try 50, 150, etc.
#     max_depth=10,  # Experiment with different depths like 5, 15, None
#     min_samples_split=4,  # Try different values like 2, 6, 10
#     min_samples_leaf=2,  # Experiment with 1, 3, 5
#     max_features='sqrt',  # Options: 'auto', 'sqrt', 'log2', or a fraction of total features
#     random_state=42,
#     class_weight='balanced'
# )



# First Train the classifier
# rf_classifier.fit(X_train, y_train)

# 
# Now use X_train_smote and y_train_smote for training the model
rf_classifier.fit(X_train_smote, y_train_smote)

# Analysis of Feature Importance
# Get feature importances
feature_importances = rf_classifier.feature_importances_

# Get feature names
feature_names = feature_names

# Create a DataFrame to view feature importances
importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': feature_importances})

# Sort the DataFrame by importance
importance_df = importance_df.sort_values(by='Importance', ascending=False)

# Display the sorted DataFrame
print(importance_df)


# Make predictions on the test set
predictions = rf_classifier.predict(X_test)

# Evaluate the classifier
accuracy = accuracy_score(y_test, predictions)
# Calculate F1-score
recall = recall_score(y_test, predictions, average='weighted')  # 'weighted' for class imbalance

precision = precision_score(y_test, predictions, average='weighted', zero_division=0) # 'weighted' for class imbalance
f1 = f1_score(y_test, predictions, average='weighted', zero_division=0) # 'weighted' accounts for label imbalance in multiclass classification

classification_rep = classification_report(y_test, predictions) # Generate a classification report

# Perform cross-validation
cv_scores = cross_val_score(rf_classifier, X, y, cv=5)

# Output results
print(f'Accuracy: {accuracy}')
print(f'Precision: {precision}')
print(f'Recall: {recall}')
print(f'F1-Score: {f1}')
print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean CV Score: {cv_scores.mean()}')
print(f'Classification Report:\n{classification_rep}')
