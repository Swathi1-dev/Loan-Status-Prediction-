import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pickle
import joblib
import os
import logging

# from imblearn.over_smapling import SMOTE
from dotenv import load_dotenv
from sklearn.pipeline import Pipeline


load_dotenv()

project_root = os.getenv("project_root")

# os.getenv("log_path").parent.mkdir(parents=True, exist_ok=True)
data_path = os.path.join(project_root, os.getenv("data_path"))

log_path = os.path.join(project_root, os.getenv("log_path"))

# os.makedirs(os.path.dirname(log_path), exist_ok=True)

# setting up logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_path), logging.StreamHandler()],
)


def load_data(file_path):
    try:
        data = pd.read_csv(file_path)
        logging.info("Data Loaded Successfully")
        return data
    except Exception as e:
        logging.error("Error loading data: %s", e)
        raise


data = load_data(data_path)


def preprocess_data(data):
    # Handle missing values
    data.dropna(inplace=True)
    data.reset_index(drop=True, inplace=True)

    data["Dependents"] = data["Dependents"].replace({"3+": 4})
    data["Dependents"] = data["Dependents"].astype(int)

    data.drop(columns=["Loan_ID"], inplace=True)

    data["Married"] = data["Married"].map({"Yes": 1, "No": 0})
    data["Gender"] = data["Gender"].map({"Male": 1, "Female": 0})
    data["Education"] = data["Education"].map({"Graduate": 1, "Not Graduate": 0})
    data["Self_Employed"] = data["Self_Employed"].map({"Yes": 1, "No": 0})
    data["Property_Area"] = data["Property_Area"].map(
        {"Urban": 2, "Rural": 0, "Semiurban": 1}
    )
    data["Loan_Status"] = data["Loan_Status"].map({"Y": 1, "N": 0})

    logging.info("Data preprocessed successfully")

    return data


data = preprocess_data(data)


def split_data(data):
    x = data.drop(os.getenv("target_col"), axis=1)
    y = data[os.getenv("target_col")]
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=float(os.getenv("test_size")),
        random_state=int(os.getenv("random_state")),
    )
    logging.info("Data split into train and test sets successfully")

    return x_train, x_test, y_train, y_test


x_train, x_test, y_train, y_test = split_data(data)


def train_model(x_train, y_train):
    model_lr = LogisticRegression()

    model = Pipeline([("scaler", StandardScaler()), ("model", model_lr)])
    model.fit(x_train, y_train)
    return model


model = train_model(x_train, y_train)


def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    acc = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    logging.info("Model evaluated successfully with accuracy: %f", acc)
    logging.info("Classification Report:\n%s", report)
    logging.info("Confusion Matrix:\n%s", cm)
    return acc, report, cm


acc, report, cm = evaluate_model(model, x_test, y_test)


def save_model(model, file_path):
    try:
        with open(file_path, "wb") as f:
            pickle.dump(model, f)
        logging.info("Model saved successfully at %s", file_path)
    except Exception as e:
        print("Error saving model: %s", e)
        logging.error("Error while saving model: %s", e)
        raise


# os.getenv("model_path").parent.mkdir(parents=True, exist_ok=True)


save_model(model, os.getenv("model_path"))


logging.info("Model training pipeline completed successfully")
