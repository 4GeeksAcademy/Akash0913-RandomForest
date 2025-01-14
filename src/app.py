import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Load dataset and remove duplicates
total_data = pd.read_csv("https://raw.githubusercontent.com/4GeeksAcademy/decision-tree-project-tutorial/main/diabetes.csv")
total_data = total_data.drop_duplicates().reset_index(drop=True)

# Separate features (X) and target variable (y)
X = total_data.drop("Outcome", axis=1)
y = total_data["Outcome"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature selection: Select top 7 features
selection_model = SelectKBest(k=7)
selection_model.fit(X_train, y_train)

# Extract selected feature names and transform the data
selected_columns = X_train.columns[selection_model.get_support()]
X_train_sel = pd.DataFrame(selection_model.transform(X_train), columns=selected_columns)
X_test_sel = pd.DataFrame(selection_model.transform(X_test), columns=selected_columns)

# Add the target variable back to the selected feature data
X_train_sel["Outcome"] = y_train.values
X_test_sel["Outcome"] = y_test.values

# Save cleaned training and testing datasets to CSV
X_train_sel.to_csv("clean_train.csv", index=False)
X_test_sel.to_csv("clean_test.csv", index=False)

# Load the cleaned datasets
train_data = pd.read_csv("clean_train.csv")
test_data = pd.read_csv("clean_test.csv")

# Separate features and target variable for the model
X_train = train_data.drop(["Outcome"], axis=1)
y_train = train_data["Outcome"]
X_test = test_data.drop(["Outcome"], axis=1)
y_test = test_data["Outcome"]

# Initialize and train the Random Forest Classifier
model = RandomForestClassifier(n_estimators=60, random_state=42)
print(model.fit(X_train, y_train))

# Predict outcomes for the test data
y_pred = model.predict(X_test)

# Print predictions and model accuracy
print(y_pred)
print(accuracy_score(y_test, y_pred))