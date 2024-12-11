import joblib

# Load the model
pipeline = joblib.load('models/my_model.joblib')

# Test it
test_text = "This is a test news article"
try:
    pred = pipeline.predict_proba([test_text])
    print("Pipeline works! Prediction:", pred)
except Exception as e:
    print("Error:", e)