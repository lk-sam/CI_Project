from joblib import dump, load
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV, train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier


def load_and_preprocess_data():
    le = preprocessing.LabelEncoder()

    df = pd.read_csv('train.csv')
    train_df, test_df = train_test_split(df, test_size=0.25, random_state=0)  # 0.25 means 25% of data will be for testing, 75% for training.

    train_df["ethnicity"][train_df["ethnicity"].isin({"Pasifika", "Hispanic", "Turkish", "?"})] = "Others"
    test_df["ethnicity"][test_df["ethnicity"].isin({"Pasifika", "Hispanic", "Turkish", "?"})] = "Others"

    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            train_df[col] = le.fit_transform(train_df[col])
            
    for col in test_df.columns:
        if test_df[col].dtype == 'object':
            test_df[col] = le.fit_transform(test_df[col])

    return train_df, test_df

def test_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print('Accuracy:', accuracy)

def visualize_data(train_df):
    plt.figure(figsize=(18,8))
    df = train_df.corr()
    mask = np.triu(np.ones_like(df))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df, annot=True, cbar=False, cmap="Blues",mask=mask)
    plt.show()

def train_model(X_train, y_train):
    le = preprocessing.LabelEncoder()

    X_train["ethnicity"][X_train["ethnicity"].isin({"Pasifika", "Hispanic", "Turkish", "?"})] = "Others"
    for col in X_train.columns:
        if X_train[col].dtype == 'object':
            X_train[col] = le.fit_transform(X_train[col])

    

    

    # Extra Tree Classifier
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    param_grid = {'n_estimators': [50, 150, 200, 250, 300, 500, 1000],'max_depth': [2, 4, 6, 8, 10]}
    model_xt = ExtraTreesClassifier(random_state=0)
    grid_model = GridSearchCV(model_xt,param_grid,cv=kf)
    grid_model.fit(X_train, y_train)

    # Logistic Regression
    param_grid={"C":np.logspace(-3,3,10), "penalty":["l1","l2"]}
    model_lr = LogisticRegression(solver='saga', tol=1e-5, max_iter=10000, random_state=0)
    grid_model_lr = GridSearchCV(model_lr,param_grid,cv=kf)
    grid_model_lr.fit(X_train, y_train)

    # Multi-layer Perceptron (MLP)
    param_grid_mlp = {
    'hidden_layer_sizes': [(25, 50, 25), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.001, 0.01, 0.05, 0.1],
    'learning_rate': ['constant','adaptive'],
    }

    model_mlp = MLPClassifier(max_iter=1000, random_state=0)
    grid_model_mlp = GridSearchCV(model_mlp, param_grid_mlp, cv=kf)
    grid_model_mlp.fit(X_train, y_train)

    return grid_model, grid_model_lr, grid_model_mlp

def make_prediction(input_data):
    # Load the model
    model_xt = load('extra_tree_model.joblib')
    model_lr = load('logistic_regression_model.joblib')
    model_mlp = load('mlp_model.joblib')

    # Prepare the LabelEncoder
    le = LabelEncoder()

    # Convert 'yes'/'no' to 1/0 for 'autism'
    input_data[10] = 1 if input_data[10] == 'yes' else 0

    # Convert ethnicity to numerical form
    input_data[12] = le.fit_transform([input_data[12]])[0]

    # Predict
    preds_xt = model_xt.predict_proba([input_data])
    preds_lr = model_lr.predict_proba([input_data])
    preds_mlp = model_mlp.predict_proba([input_data])

    # Ensembling LR and XTRA
    new_preds = preds_lr[:,1]*0.3 + preds_xt[:,1]*0.3 + preds_mlp[:,1]*0.4

    return new_preds[0]

def main():
    # Load and preprocess the data
    train_df, test_df = load_and_preprocess_data()

    X_train = train_df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim','result','ethnicity']]
    y_train = train_df['Class/ASD']

    X_test = test_df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim','result','ethnicity']]
    y_test = test_df['Class/ASD']

    # Train the model and get the trained models
    grid_model, grid_model_lr, grid_model_mlp = train_model(X_train, y_train)

    # Save the trained models
    dump(grid_model, 'extra_tree_model.joblib')
    dump(grid_model_lr, 'logistic_regression_model.joblib')
    dump(grid_model_mlp, 'mlp_model.joblib')

    # Test the models
    print("Testing Extra Trees Classifier")
    test_model(grid_model, X_test, y_test)

    print("Testing Logistic Regression")
    test_model(grid_model_lr, X_test, y_test)

    print("Testing MLP")
    test_model(grid_model_mlp, X_test, y_test)

if __name__ == "__main__":
    main()