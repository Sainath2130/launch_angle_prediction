import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the KNN model
with open('models\knn_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Create a Streamlit app
st.title("Launch Angle Predictor")

# Create a sidebar with input fields
st.sidebar.header("Input Parameters")
exit_velocity = st.sidebar.number_input("Exit Velocity (mph)", min_value=0.0, value=80.0)
hit_distance = st.sidebar.number_input("Hit Distance (ft)", min_value=0.0, value=200.0)

# Create a button to trigger the prediction
if st.sidebar.button("Predict Launch Angle"):
    # Create a numpy array from the input values
    input_data = np.array([[exit_velocity, hit_distance]])

    # Use the KNN model to make a prediction
    prediction = model.predict(input_data)

    # Display the predicted launch angle
    st.header("Predicted Launch Angle")
    st.write(f"The predicted launch angle is: {prediction[0]:.2f} degrees")

    # Display a chart to visualize the prediction
    st.header("Launch Angle Visualization")
    fig, ax = plt.subplots()
    ax.plot([exit_velocity], [prediction[0]], 'ro')
    ax.set_xlabel('Exit Velocity (mph)')
    ax.set_ylabel('Launch Angle (degrees)')
    ax.set_title('Launch Angle vs. Exit Velocity')
    st.pyplot(fig)

    # Display a table to compare the prediction with other values
    st.header("Launch Angle Comparison")
    data = {
        'Exit Velocity (mph)': [80, 90, 100],
        'Hit Distance (ft)': [200, 250, 300],
        'Launch Angle (degrees)': [model.predict(np.array([[80, 200]])), model.predict(np.array([[90, 250]])), model.predict(np.array([[100, 300]]))]
    }
    df = pd.DataFrame(data)
    st.write(df)