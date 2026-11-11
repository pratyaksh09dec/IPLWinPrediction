import pandas as pd
import numpy as np
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print("=" * 60)
print("IPL WIN PROBABILITY MODEL TRAINING")
print("=" * 60)

df = pd.read_csv("Cleaned_IPL_Data.csv")

print("\nDataset Loaded Successfully\n")

print(df.head())

print("\nDataset Shape :", df.shape)

print("\nDataset Information\n")

print(df.info())

print("\nMissing Values\n")

print(df.isnull().sum())

df = df.dropna()

df = df.drop_duplicates()

df = df.reset_index(drop=True)

print("\nDataset Shape After Cleaning :", df.shape)

print("\nClass Distribution\n")

print(df["result"].value_counts())

print("\nClass Distribution (%)\n")

print(df["result"].value_counts(normalize=True) * 100)

X = df.drop(columns="result")

y = df["result"]

print("\nFeatures\n")

print(X.head())

print("\nTarget\n")

print(y.head())

X = pd.get_dummies(
    X,
    columns=[
        "batting_team",
        "bowling_team",
        "city"
    ]
)

print("\nShape After Encoding :", X.shape)

model_columns = X.columns

pickle.dump(
    model_columns,
    open(
        "model_columns.pkl",
        "wb"
    )
)

metadata = {
    "batting_teams": sorted(df["batting_team"].unique()),
    "bowling_teams": sorted(df["bowling_team"].unique()),
    "venues": sorted(df["city"].unique())
}

pickle.dump(
    metadata,
    open(
        "metadata.pkl",
        "wb"
    )
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining Samples :", X_train.shape)

print("Testing Samples :", X_test.shape)


rf_model = RandomForestClassifier(
    n_estimators=300,
    max_depth=12,
    random_state=42,
    n_jobs=-1
)

rf_model.fit(
    X_train,
    y_train
)

y_pred = rf_model.predict(X_test)

y_prob = rf_model.predict_proba(X_test)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy :", round(accuracy * 100, 2), "%")

print("\nClassification Report\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)

importance = pd.DataFrame(
    {
        "Feature": X.columns,
        "Importance": rf_model.feature_importances_
    }
)

importance = importance.sort_values(
    by="Importance",
    ascending=False
)

print(importance.head(20))

pickle.dump(
    rf_model,
    open(
        "IPL_Win_Probability_Model.pkl",
        "wb"
    )
)

loaded_model = pickle.load(
    open(
        "IPL_Win_Probability_Model.pkl",
        "rb"
    )
)

loaded_columns = pickle.load(
    open(
        "model_columns.pkl",
        "rb"
    )
)

sample = X.iloc[[0]]

prediction = loaded_model.predict(sample)

probability = loaded_model.predict_proba(sample)

print("\nSample Prediction")

print("Predicted Result :", prediction[0])

print("Lose Probability :", round(probability[0][0] * 100, 2), "%")

print("Win Probability :", round(probability[0][1] * 100, 2), "%")