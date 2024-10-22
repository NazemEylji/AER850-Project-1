import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import f1_score, accuracy_score, precision_score, confusion_matrix
import joblib

# Step 1: Data Processing
file_path = 'D:\AER850\AER850-Project-1\Project_1_Data.csv'
data = pd.read_csv(file_path)
print("Step 1: Data loaded successfully.")
print(data.head())

# Step 2: Data Visualization
# Summary statistics for each feature
summary_stats = data.describe()
print("Step 2: Summary Statistics:")
print(summary_stats)

# Create separate visualizations for the dataset, adjusting figure size
plt.figure(figsize=(8, 6))
sns.histplot(data['X'], bins=20, color='blue', kde=True)  # kde=True adds the curve of best fit
plt.title('Distribution of X with KDE')
plt.xlabel('X-coordinate')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['Y'], bins=20, color='green', kde=True)
plt.title('Distribution of Y with KDE')
plt.xlabel('Y-coordinate')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
sns.histplot(data['Z'], bins=20, color='red', kde=True)
plt.title('Distribution of Z with KDE')
plt.xlabel('Z-coordinate')
plt.ylabel('Frequency')
plt.tight_layout()
plt.show()

# Step 3: Correlation Analysis
correlation_matrix = data.corr()
print("Step 3: Correlation Matrix:")
print(correlation_matrix)

# Plotting heatmap for the correlation matrix with adjusted size and layout
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title("Correlation Matrix")
plt.tight_layout()
plt.show()

# Step 4: Classification Model Development/Engineering
X = data[['X', 'Y', 'Z']]
y = data['Step']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define models
models = {
    "RandomForest": RandomForestClassifier(),
    "SVM": SVC(),
    "KNN": KNeighborsClassifier()
}

# Hyperparameter grids
param_grids = {
    "RandomForest": {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20], 'min_samples_split': [2, 5, 10]},
    "SVM": {'C': [0.1, 1, 10], 'kernel': ['linear', 'rbf'], 'gamma': ['scale', 'auto']},
    "KNN": {'n_neighbors': [3, 5, 7], 'weights': ['uniform', 'distance']}
}

# Perform GridSearchCV for each model
best_models = {}
for model_name, model in models.items():
    grid_search = GridSearchCV(model, param_grids[model_name], cv=5, scoring='accuracy')
    grid_search.fit(X_train, y_train)
    best_models[model_name] = grid_search.best_estimator_
    print(f"Step 4: Best {model_name} model: {best_models[model_name]}")

# RandomizedSearchCV for RandomForest
param_dist = {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_split': [2, 5, 10]}
random_search = RandomizedSearchCV(RandomForestClassifier(), param_distributions=param_dist, n_iter=10, cv=5, scoring='accuracy')
random_search.fit(X_train, y_train)
best_random_forest = random_search.best_estimator_
print(f"Step 4: Best RandomForest model from RandomizedSearch: {best_random_forest}")

# Step 5: Model Performance Analysis
def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    cm = confusion_matrix(y_test, y_pred)
    
    # Plot confusion matrix
    plt.figure(figsize=(6, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap="Blues")
    plt.title(f"{model_name} - Confusion Matrix")
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()
    
    return accuracy, precision, f1, cm

performance_metrics = {}
for model_name, model in best_models.items():
    accuracy, precision, f1, cm = evaluate_model(model, X_test, y_test, model_name)
    performance_metrics[model_name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'F1 Score': f1,
        'Confusion Matrix': cm
    }
    print(f"Step 5: {model_name} Performance:\nAccuracy: {accuracy}\nPrecision: {precision}\nF1 Score: {f1}\n")

# Evaluate the RandomizedSearchCV model
accuracy, precision, f1, cm = evaluate_model(best_random_forest, X_test, y_test, "RandomizedSearch_RandomForest")
performance_metrics["RandomizedSearch_RandomForest"] = {
    'Accuracy': accuracy,
    'Precision': precision,
    'F1 Score': f1,
    'Confusion Matrix': cm
}
print(f"Step 5: RandomizedSearch RandomForest Performance:\nAccuracy: {accuracy}\nPrecision: {precision}\nF1 Score: {f1}\n")

# Step 6: Stacked Model Performance Analysis
estimators = [
    ('knn', best_models['KNN']),
    ('svm', best_models['SVM'])
]
stacking_clf = StackingClassifier(estimators=estimators, final_estimator=RandomForestClassifier())
stacking_clf.fit(X_train, y_train)

accuracy, precision, f1, cm = evaluate_model(stacking_clf, X_test, y_test, "Stacked_Model")
performance_metrics["Stacked_Model"] = {
    'Accuracy': accuracy,
    'Precision': precision,
    'F1 Score': f1,
    'Confusion Matrix': cm
}
print(f"Step 6: Stacked Model Performance:\nAccuracy: {accuracy}\nPrecision: {precision}\nF1 Score: {f1}\n")

# Step 7: Model Evaluation
model_filename = '"D:/AER850/AER850-Project-1/stacked_model.joblib'
joblib.dump(stacking_clf, model_filename)
print(f"Step 7: Stacked Model saved as {model_filename}")

# Test the model on the provided coordinates
test_coordinates = pd.DataFrame([[9.375, 3.0625, 1.51], [6.995, 5.125, 0.3875], [0, 3.0625, 1.93], [9.4, 3, 1.8], [9.4, 3, 1.3]], columns=['X', 'Y', 'Z'])
predictions = stacking_clf.predict(test_coordinates)
print(f"Step 7: Predictions on provided coordinates: {predictions}")