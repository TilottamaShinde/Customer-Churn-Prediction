from pandas.core.internals.blocks import external_values

from src.data_preprocessing import load_data, preprocess_data
from src.model_training import train_model, evaluate_model, save_model
from src.predict import load_model, predict_churn

df = load_data('data/telco.csv')
print(df['Churn'].value_counts())
x_train, x_test, y_train, y_test = preprocess_data(df)

print(y_train.value_counts())

model = train_model(x_train, y_train)
evaluate_model(model, x_test, y_test)

save_model(model, 'model.plk')
loaded_model = load_model('model.plk')

new_data = x_test[:5]
predictions = predict_churn(loaded_model, new_data)

print("Predictions on sample data ", predictions)