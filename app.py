import streamlit as st
import pickle
import numpy as np

# Load trained model
model = pickle.load(open("model.pkl", "rb"))

st.title("🏠 House Price Prediction")

# Input fields (now 13 features)
fields = [
    ("Crime rate per capita", 0.1, "Per capita crime rate by town"),
    ("Residential land zoned (%)", 12.0, "Proportion of residential land zoned for large lots (in %)"),
    ("Non-retail business acres", 10.0, "Proportion of non-retail business acres per town"),
    ("Bounds river (1=Yes, 0=No)", 0, "Is the property located next to a river? Enter 1 for Yes, 0 for No"),
    ("Nitric oxide concentration", 0.5, "Nitric oxide concentration in air (parts per 10 million)"),
    ("Average rooms per dwelling", 6.0, "Average number of rooms per house"),
    ("% built before 1940", 60.0, "Percentage of houses built before 1940"),
    ("Distance to employment centers", 4.0, "Weighted distance to 5 major employment centers"),
    ("Highway accessibility index", 5.0, "Index of accessibility to radial highways"),
    ("Property tax rate", 300.0, "Full-value property tax rate per $10,000"),
    ("Pupil-teacher ratio", 18.0, "Pupil–teacher ratio in town’s schools"),
    ("B index", 350.0, "1000(Bk - 0.63)^2, where Bk is proportion of Black residents"),
    ("% lower status population", 12.0, "Percentage of lower status population in the area"),
]

values = []
for label, default, help_text in fields:
    if isinstance(default, int):
        values.append(st.number_input(label, value=default, step=1, help=help_text))
    else:
        values.append(st.number_input(label, value=default, help=help_text))

# Prediction
if st.button("Predict Price"):
    prediction = model.predict([values])[0]
    price_in_inr = prediction * 1000 * 83  # Convert $1000s to INR
    st.success(f"Estimated Price: ₹{price_in_inr:,.0f}")
