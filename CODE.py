import pandas as pd
import numpy as np
import tkinter as tk
from tkinter import scrolledtext
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
import joblib
import threading
import warnings
warnings.filterwarnings('ignore')

class MLApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Machine Learning Model Training")
        self.root.geometry("800x600")

        # Create a text area for displaying results
        self.result_text = scrolledtext.ScrolledText(self.root, wrap=tk.WORD, height=20, width=80)
        self.result_text.pack(padx=10, pady=10)

        # Create a button to start the model training
        self.train_button = tk.Button(self.root, text="Train Models", command=self.start_training)
        self.train_button.pack(pady=10)

    def start_training(self):
        # Run the model training in a separate thread
        thread = threading.Thread(target=self.train_models)
        thread.start()

    def train_models(self):
        self.result_text.delete(1.0, tk.END)  # Clear previous results

        # Generate Synthetic Dataset
        X, y = make_classification(n_samples=1000, n_features=20, n_informative=15, n_redundant=5, random_state=42)
        df = pd.DataFrame(X, columns=[f'Feature_{i}' for i in range(X.shape[1])])
        df['target'] = y

        # Data Preprocessing
        scaler = StandardScaler()
        X = df.drop('target', axis=1)
        y = df['target']
        X_scaled = scaler.fit_transform(X)

        # Split Data into Training and Testing Sets
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

        # Model Selection and Training
        models = {
            'Logistic Regression': LogisticRegression(),
            'K-Nearest Neighbors': KNeighborsClassifier(),
            'Support Vector Classifier': SVC(),
            'Decision Tree': DecisionTreeClassifier(),
            'Random Forest': RandomForestClassifier()
        }

        results = {}
        for model_name, model in models.items():
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            accuracy = accuracy_score(y_test, y_pred)
            results[model_name] = accuracy
            self.result_text.insert(tk.END, f"{model_name} Accuracy: {accuracy:.2f}\n")
            self.result_text.insert(tk.END, "Classification Report:\n")
            self.result_text.insert(tk.END, classification_report(y_test, y_pred) + "\n")

        # Hyperparameter Tuning for Random Forest
        self.result_text.insert(tk.END, "Hyperparameter Tuning for Random Forest:\n")
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        grid_search = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring='accuracy', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        best_rf_model = grid_search.best_estimator_

        # Model Evaluation
        self.result_text.insert(tk.END, "\nBest Random Forest Model after Hyperparameter Tuning:\n")
        self.result_text.insert(tk.END, f"Best Parameters: {grid_search.best_params_}\n")
        y_pred_best_rf = best_rf_model.predict(X_test)
        self.result_text.insert(tk.END, f"Accuracy: {accuracy_score(y_test, y_pred_best_rf)}\n")
        self.result_text.insert(tk.END, "Confusion Matrix:\n")
        self.result_text.insert(tk.END, str(confusion_matrix(y_test, y_pred_best_rf)) + "\n")

        # Model Selection
        best_model_name = max(results, key=results.get)
        self.result_text.insert(tk.END, f"\nThe best model is: {best_model_name} with accuracy {results[best_model_name]:.2f}\n")

        # Save the Best Model
        joblib.dump(best_rf_model, 'best_model.pkl')
        self.result_text.insert(tk.END, "\nBest model saved as 'best_model.pkl'. You can use this model for predictions.\n")

# Create the Tkinter root window
root = tk.Tk()
app = MLApp(root)
root.mainloop()
