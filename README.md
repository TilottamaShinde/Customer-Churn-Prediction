***Customer Churn Prediction***

A machine learning project to predict whether a telecom customer is likely to churn (i.e., stop using the service). Built using Python and scikit-learn, this project uses the popular Telco Customer Churn dataset.



##  Overview

Customer churn prediction is crucial for businesses to retain customers and reduce losses. In this project, we:

- Load and clean the dataset  
- Encode categorical features  
- Scale numerical features  
- Train a **Logistic Regression** model  
- Evaluate the model using accuracy and confusion matrix  
- Predict churn on new samples

  ## 📁 Project Structure

Customer-Churn-Prediction/ │ ├── data/ │ └── telco.csv # Dataset file │ ├── src/ │ ├── data_preprocessing.py # Functions for loading and preprocessing data │ ├── model_training.py # Functions to train, evaluate, and save the model │ └── predict.py # Functions to load model and make predictions │ ├── model.plk # Saved trained model ├── main.py # Main driver script └── README.md # Project documentation

yaml
Copy
Edit

##  Getting Started

###  Clone the Repository

```bash
git clone https://github.com/TilottamaShinde/Customer-Churn-Prediction.git
cd Customer-Churn-Prediction


***Create Virtual Environment***
bash
Copy
Edit
python3 -m venv .venv
source .venv/bin/activate       # On macOS/Linux
# OR
.venv\Scripts\activate          # On Windows

### Install Dependencies
bash
Copy
Edit
pip install -r requirements.txt
pip install pandas scikit-learn

### Run the Project
python main.py

### Technologies Used
Python 3.x

pandas

scikit-learn

Jupyter Notebook / PyCharm (optional IDEs)

### Dataset
Telco Customer Churn Dataset on Kaggle

### Author
Tilottama Shinde
GitHub: @TilottamaShinde

