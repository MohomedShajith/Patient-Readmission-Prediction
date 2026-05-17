import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils.class_weight import compute_sample_weight
from sklearn.metrics import confusion_matrix,classification_report


if __name__ =="__main__":

    X_train = joblib.load('model/X_train.pkl')
    X_test = joblib.load('model/X_test.pkl')
    y_train = joblib.load('model/y_train.pkl')
    y_test = joblib.load('model/y_test.pkl')


    feature_names = X_train.columns.tolist()
    
    X_train = X_train.values
    X_test = X_test.values
    y_train = y_train.values
    y_test = y_test.values

#Logistic Regression     
    log_model = LogisticRegression(random_state=42,max_iter=1000,class_weight="balanced")
    log_model.fit(X_train,y_train)


    log_proba = log_model.predict_proba(X_test)
    log_threshold = 0.40

    log_tuned_preds = np.argmax(log_proba, axis=1)
    log_tuned_preds[log_proba[:, 0] > log_threshold] = 0

    log_tuned_cf = confusion_matrix(y_test, log_tuned_preds)
    log_tuned_cl = classification_report(y_test, log_tuned_preds)

    print('Logistic Regression')
    print(log_tuned_cf)
    print(log_tuned_cl)



#Random Forest Classifier
    Rf_model = RandomForestClassifier(n_estimators=100,random_state=42,class_weight="balanced")
    Rf_model.fit(X_train,y_train)

    Rf_proba = Rf_model.predict_proba(X_test)
    Rf_threshold = 0.20

    Rf_tuned_preds = np.argmax(Rf_proba, axis=1)
    Rf_tuned_preds[Rf_proba[:, 0] > Rf_threshold] = 0

    Rf_tuned_cf = confusion_matrix(y_test, Rf_tuned_preds)
    Rf_tuned_cl = classification_report(y_test, Rf_tuned_preds)


    print('Random Forest Classifier')
    print(Rf_tuned_cf)
    print(Rf_tuned_cl)


#XGBoost Classifier
    xg_model = XGBClassifier(
    n_estimators=200,
    max_depth=5,
    learning_rate=0.075,
    eval_metric="mlogloss",
    random_state=42
     )
   
    xg_model.fit(X_train, y_train)

    xg_proba = xg_model.predict_proba(X_test)
    xg_threshold = 0.20

    xg_tuned_preds = np.argmax(xg_proba, axis=1)
    xg_tuned_preds[xg_proba[:, 0] > xg_threshold] = 0

    feat_importance = pd.Series(xg_model.feature_importances_, index=feature_names)
    feat_importance.nlargest(15).plot(kind='barh')
    plt.title('Top 15 Important Features')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    #print(feat_importance.nlargest(15))
    #print(feat_importance.nsmallest(10))

    xg_tuned_cf = confusion_matrix(y_test, xg_tuned_preds)
    xg_tuned_cl = classification_report(y_test, xg_tuned_preds)

    print('XGBoost Classifier')

    print(xg_tuned_cf)
    print(xg_tuned_cl)

    joblib.dump(xg_model, "model/xg_model.pkl")
