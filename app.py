import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor


st.title("AI Synthetic Patient Data Generator")

st.write(
    "Generate artificial patient data using machine learning."
)


# Original dataset
real_data = pd.DataFrame({
    "Age": [25,35,45,55,65,70,30,50,60,40],
    "BloodPressure": [110,120,135,145,160,170,115,140,155,130],
    "HeartRate": [72,75,80,85,90,95,70,82,88,78],
    "Glucose": [85,95,110,130,150,170,90,125,145,105]
})


# AI model
model = RandomForestRegressor(
    n_estimators=50,
    random_state=42
)

model.fit(
    real_data[["Age","BloodPressure","HeartRate"]],
    real_data["Glucose"]
)


amount = st.slider(
    "Number of synthetic patients",
    10,
    5000,
    1000
)


if st.button("Generate Synthetic Data"):

    synthetic = pd.DataFrame({

        "Age": np.random.randint(
            20,80,amount
        ),

        "BloodPressure": np.random.randint(
            100,180,amount
        ),

        "HeartRate": np.random.randint(
            60,100,amount
        )
    })


    synthetic["Glucose"] = (
        model.predict(synthetic)
        .astype(int)
    )


    st.subheader("Generated Synthetic Patients")

    st.dataframe(synthetic)


    st.download_button(
        label="Download CSV",
        data=synthetic.to_csv(index=False),
        file_name="synthetic_patient_data.csv"
    )
