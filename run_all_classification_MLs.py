import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
import seaborn as sns
import matplotlib.pyplot as plt



def get_model_and_param_grid(model_type):
    if model_type == "LogisticRegression":
        model = LogisticRegression()
        apply_scaling = True
        param_grid = {
            'penalty': ['l1', 'l2', 'elasticnet', 'none'],
            'C': [0.01, 0.1, 1, 10, 100]
        }
    elif model_type == "DecisionTree":
        model = DecisionTreeClassifier()
        apply_scaling = False
        param_grid = {
            'criterion': ['gini', 'entropy'],
            'max_depth': [None, 10, 20, 30, 40, 50]
        }
    elif model_type == "RandomForest":
        model = RandomForestClassifier()
        apply_scaling = False
        param_grid = {
            'n_estimators': [10, 50, 100, 200],
            'criterion': ['gini'],
            'max_depth': [None, 10, 20, 30, 40, 50]
        }
    elif model_type == "KNeighbors":
        model = KNeighborsClassifier()
        apply_scaling = True
        param_grid = {
            'n_neighbors': [3, 5, 7, 9, 11],
            'weights': ['uniform', 'distance']
        }
    elif model_type == "SVM":
        model = SVC(probability=True)
        apply_scaling = True
        param_grid = {
            'C': [0.01, 0.1, 1, 10],   # 'C': [0.01, 0.1, 1, 10, 100] - will take more time
            'kernel': ['linear', 'poly', 'rbf', 'sigmoid']
        }
    else:
        raise ValueError("Invalid model type provided.")
    
    return model, param_grid, apply_scaling


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, roc_curve
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def evaluate_classification_model(df, target_var, feature_list, model_type, target_labels):
    X = df[feature_list]
    y = df[target_var]
    
    try:
        # Initialize the model and parameter grid
        model, param_grid, apply_scaling = get_model_and_param_grid(model_type)

        # Apply scaling if necessary
        if apply_scaling:
            scaler = StandardScaler()
            X = scaler.fit_transform(X)
        
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Perform GridSearchCV
        grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy')
        grid_search.fit(X_train, y_train)
        
        # Get the best estimator
        optimized_clf = grid_search.best_estimator_
        
        # Predictions
        train_preds = optimized_clf.predict_proba(X_train)[:, 1]
        test_preds = optimized_clf.predict_proba(X_test)[:, 1]
        train_class_preds = optimized_clf.predict(X_train)
        test_class_preds = optimized_clf.predict(X_test)
        
        # Accuracy
        train_accuracy = accuracy_score(y_train, train_class_preds)
        test_accuracy = accuracy_score(y_test, test_class_preds)
        
        # Other Metrics
        train_precision = precision_score(y_train, train_class_preds)
        test_precision = precision_score(y_test, test_class_preds)
        train_recall = recall_score(y_train, train_class_preds)
        test_recall = recall_score(y_test, test_class_preds)
        train_f1 = f1_score(y_train, train_class_preds)
        test_f1 = f1_score(y_test, test_class_preds)
        test_roc_auc = roc_auc_score(y_test, test_preds)
        
        print(f"Model: {model_type}")
        print("The accuracy on train data is ", train_accuracy)
        print("The accuracy on test data is ", test_accuracy)
        print("The precision on train data is ", train_precision)
        print("The precision on test data is ", test_precision)
        print("The recall on train data is ", train_recall)
        print("The recall on test data is ", test_recall)
        print("The f1 on train data is ", train_f1)
        print("The f1 on test data is ", test_f1)
        print("The roc_auc_score on test data is ", test_roc_auc)
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, test_class_preds)
        tn, fp, fn, tp = cm.ravel()
        cm_labels = np.array([[f'TN\n{tn}', f'FP\n{fp}'], [f'FN\n{fn}', f'TP\n{tp}']])
        
        print(f'Confusion Matrix:\n{cm_labels}')
        
        plt.figure(figsize=(10, 7))
        sns.heatmap(cm, annot=cm_labels, fmt='', cmap='Blues', xticklabels=[f'Predicted: {target_labels[0]}', f'Predicted: {target_labels[1]}'], yticklabels=[f'Actual: {target_labels[0]}', f'Actual: {target_labels[1]}'])
        plt.xlabel('Predicted labels')
        plt.ylabel('True labels')
        plt.title(f'Confusion Matrix for {model_type}')
        plt.show()
        
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, test_preds)
        plt.figure(figsize=(10, 7))
        plt.plot(fpr, tpr, label=f'{model_type} (AUC = {test_roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend(loc='best')
        plt.show()
        
        # Return metrics and test predictions for ROC curve
        return {
            'model_type': model_type,
            'train_accuracy': train_accuracy,
            'test_accuracy': test_accuracy,
            'train_precision': train_precision,
            'test_precision': test_precision,
            'train_recall': train_recall,
            'test_recall': test_recall,
            'train_f1': train_f1,
            'test_f1': test_f1,
            'test_roc_auc': test_roc_auc,
            'test_preds': test_preds,  # Add predicted probabilities for ROC curve
            'y_test': y_test  # Add y_test for ROC curve
        }
    except ValueError as e:
        print(e)
        return None

def evaluate_classification_models(models, df, target_var, feature_list, target_labels):
    results = []

    for model in models:
        result = evaluate_classification_model(df, target_var, feature_list, model, target_labels)
        if result is not None:
            results.append(result)

    # Convert results to a DataFrame for easier plotting
    metrics_df = pd.DataFrame(results)

    # Plot the metrics comparison for all models
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        sns.barplot(x='model_type', y=f'test_{metric}', data=metrics_df)
        plt.title(f'Test {metric.capitalize()} Comparison')
        plt.ylabel(f'Test {metric.capitalize()}')
        plt.xlabel('Model Type')
        plt.show()

    # Plot ROC AUC curves for all models
    plt.figure(figsize=(10, 8))
    for result in results:
        model_type = result['model_type']
        fpr, tpr, _ = roc_curve(result['y_test'], result['test_preds'])  # Use y_test and test_preds
        plt.plot(fpr, tpr, label=f"{model_type} (AUC = {result['test_roc_auc']:.2f})")
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC AUC Curve')
    plt.legend(loc='best')
    plt.show()





'''
target_var = 'def_pay'
feature_list = ['LIMIT_BAL', 'SEX', 'EDUCATION', 'MARRIAGE','AGE',\
                'PAY_1','PAY_2','PAY_3','PAY_4', 'PAY_5','PAY_6', \
                'BILL_AMT1','BILL_AMT2','BILL_AMT3','BILL_AMT4','BILL_AMT5','BILL_AMT6', \
                'PAY_AMT1','PAY_AMT2', 'PAY_AMT3','PAY_AMT4','PAY_AMT5','PAY_AMT6',  ]
target_lables = ['Yes','No']

models = ['LogisticRegression', 'DecisionTree', 'RandomForest', 'KNeighbors', 'SVM']
#models = ['LogisticRegression', 'DecisionTree','KNeighbors']
evaluate_classification_models(models, defaulters, target_var, feature_list,target_lables)
'''
