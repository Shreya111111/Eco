from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
import tensorflow as tf
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from flask import Flask, render_template, request, redirect, url_for, session, flash
import firebase_admin
from firebase_admin import credentials, auth
from firebase_admin import auth



# Print TensorFlow version for debugging purposes
print("TensorFlow version:", tf.__version__)

# Load the crop recommendation model
loaded_crop_model = pickle.load(open("crop_model.pkl", 'rb'))

# Attempt to load the soil classification model without compilation
try:
    soil_model = load_model('soil_model.h5', compile=False)
except Exception as e:
    print(f"Error loading soil classification model: {e}")

    # If loading fails, try to rebuild the model with a new input layer
    try:
        # Rebuild the model by adding a new input layer
        inputs = Input(shape=(150, 150, 3))  # Same shape as original input
        base_model = load_model('soil_model.h5', compile=False)
        x = base_model(inputs)  # Pass the new input to the rest of the model layers
        soil_model = Model(inputs=inputs, outputs=x)

        # Compile the model (optional, depending on your use case)
        soil_model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        print("Successfully rebuilt the soil model.")
    except Exception as rebuild_error:
        print(f"Failed to rebuild the soil model: {rebuild_error}")

# Define the class names for soil types
soil_class_names = ['Alluvial soil', 'Black Soil', 'Clay soil', 'Red soil']

# Initialize the Flask app
app = Flask(__name__)
app.secret_key = '3529e87192f6eeb47c4227990d9a48c9'

cred = credentials.Certificate('ecostore-117ae-firebase-adminsdk-uh014-592ab916cd.json')
firebase_admin.initialize_app(cred)
# Define the main route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for login page
@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.get_user_by_email(email)  # Ensure this function works
            # Optionally check password here
            session['user'] = email
            print(f'User {email} logged in successfully.')  # Debug print
            return redirect(url_for('home'))
        except Exception as e:
            print(f'Login failed: {str(e)}')  # Debug print
            flash('Login failed. Check your email and password.')
            return redirect(url_for('login'))
    return render_template('login.html')

# Route for signup page
@app.route('/signup', methods=['GET', 'POST'])
def signup():
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        try:
            user = auth.create_user(email=email, password=password)
            session['user'] = email
            flash('Signup successful. You can now log in.')
            return redirect(url_for('login'))
        except Exception as e:
            flash('Signup failed. Please try again.')
            return redirect(url_for('signup'))
    return render_template('signup.html')

# Route for logout
@app.route('/logout')
def logout():
    session.pop('user', None)
    flash('You have been logged out.')
    return redirect(url_for('login'))

# Crop Recommendation System
@app.route('/crop-recommendation', methods=['GET', 'POST'])
def crop_recommendation():
    if request.method == 'POST':
        try:
            # Get inputs from form
            N = float(request.form['N'])
            P = float(request.form['P'])
            K = float(request.form['K'])
            temperature = float(request.form['temperature'])
            humidity = float(request.form['humidity'])
            pH = float(request.form['pH'])
            rainfall = float(request.form['rainfall'])

            # Predict using the crop recommendation model
            input_value = np.array([[N, P, K, temperature, humidity, pH, rainfall]])
            prediction = loaded_crop_model.predict(input_value)
            pred = prediction[0]

            # Mapping prediction to crop names
            crops = {
                1: "Rice", 2: "Maize", 3: "Jute", 4: "Cotton", 5: "Coconut", 6: "Papaya", 7: "Orange",
                8: "Apple", 9: "Muskmelon", 10: "Watermelon", 11: "Grapes", 12: "Mango", 13: "Banana",
                14: "Pomegranate", 15: "Lentil", 16: "Blackgram", 17: "Mungbean", 18: "Mothbeans",
                19: "Pigeonpeas", 20: "Kidneybeans", 21: "Chickpea", 22: "Coffee"
            }

            result = crops.get(pred, "Sorry, we could not determine the best crop to be cultivated with the provided data.")
            return render_template('crop_recommendation.html', result=result)

        except Exception as e:
            return f"Error: {str(e)}"
    return render_template('crop_recommendation.html')

# Soil Type Prediction System
@app.route('/predict-soil-type', methods=['GET', 'POST'])
def predict_soil_type():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400

        file = request.files['image']

        if file.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Save the uploaded image to a temporary location
        img_path = os.path.join('uploads', file.filename)
        file.save(img_path)

        # Preprocess the image and make prediction
        processed_image = preprocess_image(img_path)
        predictions = soil_model.predict(processed_image)
        predicted_class_index = np.argmax(predictions[0])
        predicted_class_name = soil_class_names[predicted_class_index]

        # Remove the temporary file
        os.remove(img_path)

        return jsonify({'predicted_class': predicted_class_name})

    return render_template('soil.html')

# Helper function to preprocess the soil image
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0  # Normalize the image
    return img_array

if __name__ == '__main__':
    # Ensure the 'uploads' directory exists
    if not os.path.exists('uploads'):
        os.makedirs('uploads')

    app.run(debug=True)
