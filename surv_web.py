import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

st.set_option('deprecation.showPyplotGlobalUse', False)

st.header('CANCERVIVE: Cancer Survival Prediction')

st.write("""
This app predicts the **cancer patients' survival**!
""")

def user_input_features():
    cancer_types = [
        "Liver",
        "Bladder",
        "Lung",
        "Breast",
        "Oesophagus"
    ]
    genders = [
        "Persons",
        "Female",
        "Male"
    ]
    ages = [
        "15-44",
        "45-54",
        "55-64",
        "65-74",
        "75-99",
        "All ages"
    ]
    stages = [
        "All stages combined",
        "1",
        "2",
        "3",
        "4"
    ]
    years_options = [
        "1",
        "2",
        "3",
        "4",
        "5"
    ]

    cancer_type = st.selectbox("Cancer Type", cancer_types)
    gender = st.selectbox("Gender", genders)
    age = st.selectbox("Patient's Age", ages)
    stage = st.selectbox("Cancer Stage", stages)
    selected_years = st.selectbox("Years Since Diagnosis", years_options)

    data = {'Cancer type': cancer_type, 'Gender': gender, 'Stage': stage, 'Age At Diagnosis': age,
            'Years Since Diagnosis': selected_years}

    features = pd.DataFrame(data, index=[0])

    return features

# Add the input form
input_features = user_input_features()

# Load the Cancer survival dataset from CSV
url = 'https://raw.githubusercontent.com/sarah-liya/survival-web/main/FYP%20Cancer%20Survival%20new.csv'
cancerS_df = pd.read_csv(url, low_memory=False)

# Encode the categorical features in the original dataset
encode = ['Cancer type', 'Gender', 'Stage', 'Age At Diagnosis']
for col in encode:
    label_encoder = LabelEncoder()
    cancerS_df[col] = label_encoder.fit_transform(cancerS_df[col])
    input_features[col] = label_encoder.transform(input_features[col])

features = ['Cancer type', 'Gender', 'Stage', 'Age At Diagnosis', 'Years Since Diagnosis']
X = cancerS_df[features]
y = cancerS_df['Survival (%)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build Decision Tree Regressor
dtree = DecisionTreeRegressor()
dtree.fit(X_train, y_train)

if input_features is not None:

    # Add a button to trigger the prediction
    if st.button('Predict Cancer Survival'):
        # Predict survival
        prediction = dtree.predict(input_features)[0]

        # Display the prediction
        st.write('### Patients Survival Rate Prediction')
        st.write(f'{prediction:.2f}%')



        # Extract the attribute weights (feature importances)
        attribute_weights = dtree.feature_importances_
        # Create a DataFrame to store the attribute weights
        attribute_weights_df = pd.DataFrame({'Attribute': X_train.columns, 'Weight': attribute_weights})
        
        # Filter the attribute weights based on user input
        selected_attributes = input_features.columns
        filtered_attribute_weights_df = attribute_weights_df[attribute_weights_df['Attribute'].isin(selected_attributes)]
        
        # Display the attribute weights
        st.write('### Attribute Weights')
        st.write(filtered_attribute_weights_df)
        
        # Plot the attribute weights
        plt.figure(figsize=(10, 6))
        plt.bar(filtered_attribute_weights_df['Attribute'], filtered_attribute_weights_df['Weight'])
        plt.xticks(rotation=90)
        plt.xlabel('Attribute')
        plt.ylabel('Weight')
        plt.title('Attribute Weights')
        st.pyplot(plt)





        

        

    
