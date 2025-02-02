import os
import streamlit as st
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Load the KNN model
model_path = os.path.join('models', 'knn_model.pkl')
try:
    with open(model_path, 'rb') as f:
        model = pickle.load(f)
except FileNotFoundError:
    st.error("Model file not found. Please ensure that 'knn_model.pkl' exists in the 'models' directory.")
    st.stop()
except Exception as e:
    st.error(f"An error occurred while loading the model: {e}")
    st.stop()

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
    
    # Prepare data for comparison
    comparison_data = {
        'Exit Velocity (mph)': [80, 90, 100],
        'Hit Distance (ft)': [200, 250, 300],
        'Launch Angle (degrees)': []
    }
    
    # Calculate predictions for comparison
    for velocity, distance in zip(comparison_data['Exit Velocity (mph)'], comparison_data['Hit Distance (ft)']):
        launch_angle = model.predict(np.array([[velocity, distance]]))
        comparison_data['Launch Angle (degrees)'].append(launch_angle[0])

    # Create a DataFrame for the comparison
    df = pd.DataFrame(comparison_data)
    st.write(df)