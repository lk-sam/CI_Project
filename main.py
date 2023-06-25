import pandas as pd
from sklearn import preprocessing
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, GridSearchCV
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

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

def train_and_predict(train_df, test_df):
    X = train_df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim','result','ethnicity']]
    y = train_df['Class/ASD']
    
    X_test = test_df[['A1_Score', 'A2_Score', 'A3_Score', 'A4_Score', 'A5_Score', 'A6_Score', 'A7_Score', 'A8_Score', 'A9_Score', 'A10_Score', 'austim','result','ethnicity']]
    
    # Extra Tree Classifier
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    param_grid = {'n_estimators': [50, 150, 200, 250, 300, 500, 1000],'max_depth': [2, 4, 6, 8, 10]}
    model_xt = ExtraTreesClassifier(random_state=0)
    grid_model = GridSearchCV(model_xt,param_grid,cv=kf)
    grid_model.fit(X, y)
    preds_xt = grid_model.predict_proba(X_test)
    
    # Logistic Regression
    param_grid={"C":np.logspace(-3,3,10), "penalty":["l1","l2"]}
    model_lr = LogisticRegression(solver='saga', tol=1e-5, max_iter=10000, random_state=0)
    grid_model_lr = GridSearchCV(model_lr,param_grid,cv=kf)
    grid_model_lr.fit(X, y)
    preds_lr = grid_model_lr.predict_proba(X_test)
    
    # Ensembling LR and XTRA
    new_preds = preds_lr[:,1]*0.58 + preds_xt[:,1]*0.42 
    submission_rc = pd.DataFrame({'ID':test_df.ID,'Class/ASD':new_preds})
    submission_rc.to_csv('submission.csv',index=False)

def main():
    train_df, test_df = load_and_preprocess_data()
    visualize_data(train_df)
    train_and_predict(train_df, test_df)

if __name__ == "__main__":
    main()
