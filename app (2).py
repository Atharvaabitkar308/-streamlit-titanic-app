{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "id": "7bf97284-e973-4ff2-be70-ba3bd4c41409",
   "metadata": {},
   "outputs": [],
   "source": [
    "import pandas as pd\n",
    "from sklearn.impute import SimpleImputer\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.model_selection import train_test_split\n",
    "import pickle\n",
    "\n",
    "# Load the training data\n",
    "train_data = pd.read_csv(r\"C:\\Users\\athar\\OneDrive\\Desktop\\python\\Titanic_Train.csv\")\n",
    "\n",
    "# Handle missing values\n",
    "age_imputer = SimpleImputer(strategy='median')\n",
    "train_data['Age'] = age_imputer.fit_transform(train_data[['Age']])\n",
    "\n",
    "embarked_imputer = SimpleImputer(strategy='most_frequent')\n",
    "train_data['Embarked'] = embarked_imputer.fit_transform(train_data[['Embarked']]).ravel()\n",
    "\n",
    "train_data.drop(columns=['Cabin'], inplace=True)\n",
    "\n",
    "# Encode categorical variables\n",
    "train_data['Sex'] = train_data['Sex'].map({'male': 0, 'female': 1})\n",
    "train_data = pd.get_dummies(train_data, columns=['Embarked'], drop_first=True)\n",
    "\n",
    "train_data.drop(columns=['PassengerId', 'Name', 'Ticket'], inplace=True)\n",
    "\n",
    "# Define features and target variable\n",
    "X = train_data.drop(columns=['Survived'])\n",
    "y = train_data['Survived']\n",
    "\n",
    "# Split the data into training and validation sets\n",
    "X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)\n",
    "\n",
    "# Build and train the logistic regression model\n",
    "model = LogisticRegression(max_iter=1000)\n",
    "model.fit(X_train, y_train)\n",
    "\n",
    "# Save the model and imputers\n",
    "with open('logistic_regression_model.pkl', 'wb') as model_file:\n",
    "    pickle.dump(model, model_file)\n",
    "\n",
    "with open('age_imputer.pkl', 'wb') as age_file:\n",
    "    pickle.dump(age_imputer, age_file)\n",
    "\n",
    "with open('embarked_imputer.pkl', 'wb') as embarked_file:\n",
    "    pickle.dump(embarked_imputer, embarked_file)\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": 3,
   "id": "ba6203b5-c6f7-4f41-a679-a3ce2db137f1",
   "metadata": {},
   "outputs": [
    {
     "name": "stderr",
     "output_type": "stream",
     "text": [
      "2024-06-12 15:16:46.992 \n",
      "  \u001b[33m\u001b[1mWarning:\u001b[0m to view this Streamlit app on a browser, run it with the following\n",
      "  command:\n",
      "\n",
      "    streamlit run C:\\Users\\athar\\anaconda3\\Lib\\site-packages\\ipykernel_launcher.py [ARGUMENTS]\n"
     ]
    }
   ],
   "source": [
    "import streamlit as st\n",
    "import pandas as pd\n",
    "from sklearn.linear_model import LogisticRegression\n",
    "from sklearn.impute import SimpleImputer\n",
    "import pickle\n",
    "\n",
    "# Load the model and preprocessing steps\n",
    "model = pickle.load(open('logistic_regression_model.pkl', 'rb'))\n",
    "age_imputer = pickle.load(open('age_imputer.pkl', 'rb'))\n",
    "embarked_imputer = pickle.load(open('embarked_imputer.pkl', 'rb'))\n",
    "\n",
    "# Streamlit app\n",
    "st.title('Titanic Survival Prediction')\n",
    "st.write('Enter passenger details to predict survival probability.')\n",
    "\n",
    "# User input\n",
    "sex = st.selectbox('Sex', ['male', 'female'])\n",
    "age = st.number_input('Age', min_value=0, max_value=100, value=25)\n",
    "pclass = st.selectbox('Passenger Class', [1, 2, 3])\n",
    "sibsp = st.number_input('Number of Siblings/Spouses Aboard', min_value=0, max_value=10, value=0)\n",
    "parch = st.number_input('Number of Parents/Children Aboard', min_value=0, max_value=10, value=0)\n",
    "fare = st.number_input('Fare', min_value=0.0, value=15.0)\n",
    "embarked = st.selectbox('Port of Embarkation', ['C', 'Q', 'S'])\n",
    "\n",
    "# Preprocess input\n",
    "input_data = pd.DataFrame({\n",
    "    'Pclass': [pclass],\n",
    "    'Sex': [1 if sex == 'female' else 0],\n",
    "    'Age': [age],\n",
    "    'SibSp': [sibsp],\n",
    "    'Parch': [parch],\n",
    "    'Fare': [fare],\n",
    "    'Embarked_Q': [1 if embarked == 'Q' else 0],\n",
    "    'Embarked_S': [1 if embarked == 'S' else 0]\n",
    "})\n",
    "\n",
    "# Predict\n",
    "prediction_proba = model.predict_proba(input_data)[0][1]\n",
    "prediction = 'Survived' if prediction_proba >= 0.5 else 'Did not survive'\n",
    "\n",
    "st.write(f'Prediction: {prediction}')\n",
    "st.write(f'Survival Probability: {prediction_proba:.2f}')\n"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": None,
   "id": "f6d4cc33-aeba-4100-a098-0f021beeedd1",
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3 (ipykernel)",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.11.7"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 5
}
