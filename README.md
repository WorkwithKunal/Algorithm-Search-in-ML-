Machine Learning Model Training Application

Description :
- This project is a Python application with a Tkinter graphical user interface (GUI) that allows users to train and evaluate various machine learning models. The application includes features such as synthetic dataset generation, hyperparameter tuning for Random Forest, and visualization of model performance metrics.


Features :
- Synthetic Dataset Generation: Create a classification dataset with customizable features.
- Model Training and Evaluation: Train and evaluate multiple machine learning models including Logistic Regression, K-Nearest Neighbors, Support Vector Classifier, Decision Tree, and Random Forest.
- Hyperparameter Tuning: Optimize the Random Forest model using Grid Search with cross-validation.
- GUI with Tkinter: An interactive interface to start training and view results without running scripts manually.


Example Output
Here’s an example of what you can expect in the GUI:
Logistic Regression Accuracy: 0.89
Classification Report:
              precision    recall  f1-score   support

           0       0.89      0.89      0.89       103
           1       0.89      0.89      0.89       107

    accuracy                           0.89       210
   macro avg       0.89      0.89      0.89       210
weighted avg       0.89      0.89      0.89       210

...

Hyperparameter Tuning for Random Forest:
Best Parameters: {'max_depth': 20, 'min_samples_leaf': 2, 'min_samples_split': 5, 'n_estimators': 100}
Accuracy: 0.90
Confusion Matrix:
[[90  7]
 [ 8 105]]

The best model is: Random Forest with accuracy 0.90

Best model saved as 'best_model.pkl'. You can use this model for predictions.


Acknowledgments :
- Scikit-learn: For the machine learning algorithms and utilities.
- Tkinter: For providing the GUI framework.
- Python: For the programming language.
