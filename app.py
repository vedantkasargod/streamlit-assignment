import streamlit as st
import streamlit_shadcn_ui as ui
import pickle
import os

import os
import pickle
import pandas as pd
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from mlxtend.frequent_patterns import apriori, association_rules
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
 
# Ensure that the directory exists for saving models
MODEL_DIR = os.path.dirname(os.path.abspath(__file__))
 
# Load Titanic Dataset
def load_titanic_data():
    url = 'https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv'
    titanic_data = pd.read_csv(url)
    titanic_data['Sex'] = titanic_data['Sex'].map({'male': 0, 'female': 1})  # Convert categorical to numeric
    titanic_data = titanic_data.dropna(subset=['Age', 'Fare'])  # Drop rows with missing values
    return titanic_data
 
# 1. Train Linear Regression Model to predict Fare
def train_linear_regression(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Sex']]
    y = titanic_data['Fare']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = LinearRegression()
    model.fit(X_train, y_train)
 
    # Save the model
    with open(os.path.join(MODEL_DIR, 'linear_regression_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
 
    print("Linear Regression model saved.")
 
# 2. Train Logistic Regression Model to predict Survival
def train_logistic_regression(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Fare', 'Sex']]
    y = titanic_data['Survived']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Logistic Regression Accuracy: {accuracy:.4f}")
 
    # Save the model
    with open(os.path.join(MODEL_DIR, 'logistic_regression_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
 
    print("Logistic Regression model saved.")
 
# 3. Train Naive Bayes Model to predict Survival
def train_naive_bayes(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Fare', 'Sex']]
    y = titanic_data['Survived']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = GaussianNB()
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Naive Bayes Accuracy: {accuracy:.4f}")
 
    # Save the model
    with open(os.path.join(MODEL_DIR, 'naive_bayes_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
 
    print("Naive Bayes model saved.")
 
# 4. Train Decision Tree Model to predict Survival
def train_decision_tree(titanic_data):
    X = titanic_data[['Pclass', 'Age', 'Fare', 'Sex']]
    y = titanic_data['Survived']
 
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
   
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
 
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Decision Tree Accuracy: {accuracy:.4f}")
 
    # Save the model
    with open(os.path.join(MODEL_DIR, 'decision_tree_model.pkl'), 'wb') as f:
        pickle.dump(model, f)
 
    print("Decision Tree model saved.")
 
# 5. Apriori Algorithm for Recommendation System (based on Survival, Pclass, and Sex)
def train_apriori(titanic_data):
    # Use relevant binary features (1: present, 0: not present) for Apriori
    titanic_data['Survived'] = titanic_data['Survived'].apply(lambda x: 1 if x == 1 else 0)
    titanic_data['Pclass_1'] = titanic_data['Pclass'].apply(lambda x: 1 if x == 1 else 0)
    titanic_data['Pclass_2'] = titanic_data['Pclass'].apply(lambda x: 1 if x == 2 else 0)
    titanic_data['Pclass_3'] = titanic_data['Pclass'].apply(lambda x: 1 if x == 3 else 0)
   
    data_for_apriori = titanic_data[['Survived', 'Pclass_1', 'Pclass_2', 'Pclass_3', 'Sex']]
   
    # Apply the Apriori algorithm
    frequent_itemsets = apriori(data_for_apriori, min_support=0.2, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.6)
 
    print(f"Apriori Rules: \n{rules.head()}")
 
    # Save the results
    with open(os.path.join(MODEL_DIR, 'apriori_model.pkl'), 'wb') as f:
        pickle.dump(frequent_itemsets, f)
 
    print("Apriori model saved.")
 
if __name__ == "__main__":
    # Load data
    titanic_data = load_titanic_data()
 
    # Train all models
    train_linear_regression(titanic_data)
    train_logistic_regression(titanic_data)
    train_naive_bayes(titanic_data)
    train_decision_tree(titanic_data)
    train_apriori(titanic_data)

# Function to load a model from a pickle file
def load_model(model_name):
    model_path = os.path.join('models', model_name)
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Inferencing Functions
def predict_with_linear_regression(model, inputs):
    return model.predict([inputs])[0]

def predict_with_logistic_regression(model, inputs):
    return model.predict([inputs])[0]

def predict_with_naive_bayes(model, inputs):
    return model.predict([inputs])[0]

def predict_with_decision_tree(model, inputs):
    return model.predict([inputs])[0]

def load_apriori_rules():
    model_path = os.path.join('models', 'apriori_model.pkl')
    with open(model_path, 'rb') as f:
        return pickle.load(f)

# Define function to display the result
def display_prediction(prediction, description):
    st.subheader("Prediction Result")
    st.write(description)
    st.write(f"*Result:* {prediction}")

@st.fragment
def input_fragment():
    st.header("Input Features")
    Pclass = ui.select(options=[1, 2, 3], label="Pclass (Ticket Class)", key="pclass_select")
    Age = ui.slider(default_value=[25], min_value=1, max_value=100, step=1, label="Age", key="age_slider")[0]
    Fare = ui.slider(default_value=[50], min_value=0, max_value=500, step=1, label="Fare", key="fare_slider")[0]
    Sex = ui.select(options=["Male", "Female"], label="Sex", key="sex_select")
    Sex_binary = 1 if Sex == "Female" else 0
    
    inputs = [Pclass, Age, Fare, Sex_binary]
    return inputs

@st.fragment
def prediction_fragment(inputs):
    model_option = st.session_state.get('model_option', "Linear Regression (Predict Fare)")
    
    if model_option == "Linear Regression (Predict Fare)":
        st.subheader("Predict Fare using Linear Regression")
        model = load_model('linear_regression_model.pkl')
        prediction_inputs = inputs[:2] + [inputs[3]]  # Exclude Fare
        prediction = predict_with_linear_regression(model, prediction_inputs)
        display_prediction(f"${prediction:.2f}", "The predicted fare is based on the passenger's class, age, and gender.")
    
    elif model_option == "Logistic Regression (Predict Survival)":
        st.subheader("Predict Survival using Logistic Regression")
        model = load_model('logistic_regression_model.pkl')
        prediction = predict_with_logistic_regression(model, inputs)
        display_prediction("Survived" if prediction == 1 else "Did Not Survive", "The model predicts whether the passenger survived based on class, age, fare, and gender.")
    
    elif model_option == "Naive Bayes (Predict Survival)":
        st.subheader("Predict Survival using Naive Bayes")
        model = load_model('naive_bayes_model.pkl')
        prediction = predict_with_naive_bayes(model, inputs)
        display_prediction("Survived" if prediction == 1 else "Did Not Survive", "The Naive Bayes model provides a probabilistic prediction of survival based on the input features.")
    
    elif model_option == "Decision Tree (Predict Survival)":
        st.subheader("Predict Survival using Decision Tree")
        model = load_model('decision_tree_model.pkl')
        prediction = predict_with_decision_tree(model, inputs)
        display_prediction("Survived" if prediction == 1 else "Did Not Survive", "The decision tree model uses input features to classify whether the passenger survived or not.")
    
    elif model_option == "Apriori (Association Rules)":
        st.subheader("Apriori Association Rules")
        apriori_rules = load_apriori_rules()
        st.write("These are the association rules generated by the Apriori algorithm:")
        st.dataframe(apriori_rules)
        st.write("The Apriori algorithm identifies associations between survival, ticket class, and gender, showing which combinations of features tend to occur together.")

# Main UI layout
st.title("Titanic Prediction Models")

# Sidebar for navigation and model selection
with st.sidebar:
    st.title("Navigation")
    st.markdown("Choose the model and input data.")
    model_option = ui.select(
        options=[
            "Linear Regression (Predict Fare)",
            "Logistic Regression (Predict Survival)",
            "Naive Bayes (Predict Survival)",
            "Decision Tree (Predict Survival)",
            "Apriori (Association Rules)"
        ],
        label="Choose the model you want to use:",
        key="model_select"
    )
    st.session_state['model_option'] = model_option

# Call the fragment functions
inputs = input_fragment()
prediction_fragment(inputs)

# Add a button to trigger rerun
if ui.button("Rerun Prediction", key="rerun_btn"):
    st.rerun()