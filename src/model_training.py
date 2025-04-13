from scipy.stats import logistic
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib

def train_model(x_train, y_train):
    model = LogisticRegression(max_iter = 1000)
    model.fit(x_train,y_train)
    return model

def evaluate_model(model, x_test, y_test):
    y_pred = model.predict(x_test)
    print("Accuracy : ", accuracy_score(y_test, y_pred))
    print("Confusion Matrix : \n", confusion_matrix(y_test,y_pred))

def save_model(model, file_name):
    joblib.dump(model, file_name)