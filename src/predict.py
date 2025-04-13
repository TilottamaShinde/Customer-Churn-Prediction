import joblib

def load_model(filename):
    return joblib.load(filename)

def predict_churn(model, new_data):
    prediction = model.predict(new_data)
    return prediction