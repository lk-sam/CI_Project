from joblib import dump, load
import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPClassifier


def load_and_preprocess_data():
    le = preprocessing.LabelEncoder()

    train_df = pd.read_csv('train.csv')
    test_df = pd.read_csv('test.csv')

    train_df["ethnicity"][train_df["ethnicity"].isin({"Pasifika", "Hispanic", "Turkish", "?"})] = "Others"
    test_df["ethnicity"][test_df["ethnicity"].isin({"Pasifika", "Hispanic", "Turkish", "?"})] = "Others"

    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            train_df[col] = le.fit_transform(train_df[col])
            
    for col in test_df.columns:
        if test_df[col].dtype == 'object':
            test_df[col] = le.fit_transform(test_df[col])

    return train_df, test_df

def visualize_data(train_df):
    plt.figure(figsize=(18,8))
    df = train_df.corr()
    mask = np.triu(np.ones_like(df))
    plt.title('Pearson Correlation of Features', y=1.05, size=15)
    sns.heatmap(df, annot=True, cbar=False, cmap="Blues",mask=mask)
    plt.show()

def train_model(train_df):
    le = preprocessing.LabelEncoder()

    train_df["ethnicity"][train_df["ethnicity"].isin({"Pasifika", "Hispanic", "Turkish", "?"})] = "Others"
    for col in train_df.columns:
        if train_df[col].dtype == 'object':
            train_df[col] = le.fit_transform(train_df[col])

    X = train_df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim','result','ethnicity']]
    y = train_df['Class/ASD']

    

    # Extra Tree Classifier
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    param_grid = {'n_estimators': [50, 150, 200, 250, 300, 500, 1000],'max_depth': [2, 4, 6, 8, 10]}
    model_xt = ExtraTreesClassifier(random_state=0)
    grid_model = GridSearchCV(model_xt,param_grid,cv=kf)
    grid_model.fit(X, y)
    dump(grid_model, 'extra_tree_model.joblib')

    # Logistic Regression
    param_grid={"C":np.logspace(-3,3,10), "penalty":["l1","l2"]}
    model_lr = LogisticRegression(solver='saga', tol=1e-5, max_iter=10000, random_state=0)
    grid_model_lr = GridSearchCV(model_lr,param_grid,cv=kf)
    grid_model_lr.fit(X, y)
    dump(grid_model_lr, 'logistic_regression_model.joblib')

    # Multi-layer Perceptron (MLP)
    param_grid_mlp = {
    'hidden_layer_sizes': [(50,50,50), (50,100,50), (100,)],
    'activation': ['tanh', 'relu'],
    'solver': ['sgd', 'adam'],
    'alpha': [0.0001, 0.05],
    'learning_rate': ['constant','adaptive'],
    }
    model_mlp = MLPClassifier(max_iter=1000, random_state=0)
    grid_model_mlp = GridSearchCV(model_mlp, param_grid_mlp, cv=kf)
    grid_model_mlp.fit(X, y)
    dump(grid_model_mlp, 'mlp_model.joblib')

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
    train_df = pd.read_csv('train.csv')
    train_model(train_df)

if __name__ == "__main__":
    main()