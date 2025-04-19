import os
import traceback # For detailed error logging
import pandas as pd
import numpy as np
import joblib
from flask import Flask, render_template, request, redirect, url_for, flash
import google.generativeai as genai # Import Gemini library

# --- Configuration ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_FILENAME = 'cardiovascular_risk_random_forest_pipeline.joblib' # Verify this filename
MODEL_PATH = os.path.join(BASE_DIR, MODEL_FILENAME)

# --- Configure Gemini API ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    print("-----------------------------------------------------")
    print("WARNING: GEMINI_API_KEY environment variable not set.")
    print("         AI-powered suggestions will be disabled.")
    print("         Get a key from https://aistudio.google.com/app/apikey")
    print("-----------------------------------------------------")
    gemini_configured = False
else:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        gemini_configured = True
        print("Gemini API configured successfully.")
    except Exception as e:
        print(f"Error configuring Gemini API: {e}")
        gemini_configured = False


# --- Initialize Flask App ---
app = Flask(__name__)
app.secret_key = 'your_very_secret_key_part_2' # Change this

# --- Load the Model Pipeline ---
pipeline = None
expected_features = []
try:
    print(f"Attempting to load model from: {MODEL_PATH}")
    pipeline = joblib.load(MODEL_PATH)
    print("Model pipeline loaded successfully.")
    # CRITICAL: Define the feature names and order EXACTLY as used for training
    final_numerical_cols = ['age_years', 'height', 'weight', 'ap_hi', 'ap_lo', 'bmi', 'pulse_pressure']
    final_categorical_cols = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'bp_category']
    expected_features = final_numerical_cols + final_categorical_cols
    print(f"Expected features order: {expected_features}")
except FileNotFoundError:
    print(f"FATAL ERROR: Model file not found at {MODEL_PATH}")
    pipeline = None # Ensure pipeline is None if loading fails
except Exception as e:
    print(f"FATAL ERROR: An error occurred loading the model pipeline: {e}")
    traceback.print_exc()
    pipeline = None


# --- Helper Function for Blood Pressure Category (Keep as before) ---
def map_bp_to_category(ap_hi, ap_lo):
    try:
        ap_hi, ap_lo = float(ap_hi), float(ap_lo)
        if ap_hi < 120 and ap_lo < 80: return 'Normal'
        elif 120 <= ap_hi < 130 and ap_lo < 80: return 'Elevated'
        elif 130 <= ap_hi < 140 or 80 <= ap_lo < 90: return 'Hypertension Stage 1'
        elif ap_hi >= 140 or ap_lo >= 90: return 'Hypertension Stage 2+'
        else: return 'Undefined'
    except (ValueError, TypeError): return 'Undefined'


# --- Gemini Interaction Function ---
def get_gemini_suggestions(input_data, risk_score):
    """
    Calls the Gemini API to get contributing factors and suggestions.

    Args:
        input_data (dict): Dictionary containing the user's processed input features.
        risk_score (float): The predicted risk score percentage.

    Returns:
        tuple: (list_of_factors, list_of_suggestions)
               Returns default messages if API is not configured or fails.
    """
    if not gemini_configured:
        print("Gemini API not configured. Returning default suggestions.")
        return ["AI suggestions disabled."], ["Please consult a healthcare provider for personalized advice."]

    model = genai.GenerativeModel('gemini-1.5-flash-latest')

    # --- Create a clear prompt for Gemini ---
    # Map numerical codes back to readable text for the prompt
    gender_map = {1: "Female", 2: "Male"}
    cholesterol_map = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
    gluc_map = {1: "Normal", 2: "Above Normal", 3: "Well Above Normal"}
    binary_map = {0: "No", 1: "Yes"}

    prompt = f"""
    Analyze the following health data for potential cardiovascular risk factors.
    The patient's data is:
    - Age: {input_data['age_years']} years
    - Gender: {gender_map.get(input_data['gender'], 'Unknown')}
    - Height: {input_data['height']} cm
    - Weight: {input_data['weight']} kg
    - BMI: {input_data['bmi']:.1f}
    - Systolic Blood Pressure: {input_data['ap_hi']} mmHg
    - Diastolic Blood Pressure: {input_data['ap_lo']} mmHg
    - Blood Pressure Category: {input_data['bp_category']}
    - Pulse Pressure: {input_data['pulse_pressure']} mmHg
    - Cholesterol Level: {cholesterol_map.get(input_data['cholesterol'], 'Unknown')}
    - Glucose Level: {gluc_map.get(input_data['gluc'], 'Unknown')}
    - Smoker: {binary_map.get(input_data['smoke'], 'Unknown')}
    - Alcohol Intake Reported: {binary_map.get(input_data['alco'], 'Unknown')}
    - Physically Active: {binary_map.get(input_data['active'], 'Unknown')}

    An AI model predicted a cardiovascular risk score of {risk_score:.1f}%.

    Based *only* on the data provided and the risk score, act as a helpful AI assistant (NOT a doctor) and provide:
    1.  A list of the most likely **Contributing Factors** to this risk score. Be specific where possible (e.g., "High Blood Pressure (Stage 1)" instead of just "Blood Pressure"). If the profile looks generally healthy despite a score, mention overall lifestyle or age if relevant.
    2.  A list of general **Suggested Improvements** or considerations based on these factors. These should be general lifestyle suggestions (diet, exercise, smoking cessation, stress management, regular check-ups).

    **Important:**
    - DO NOT provide medical advice, diagnosis, or treatment recommendations.
    - Use cautious language (e.g., "potential factor," "consider," "general suggestion").
    - Frame suggestions positively.
    - Keep the lists concise (bullet points preferred).
    - Start the first list with the exact line: "Contributing Factors:"
    - Start the second list with the exact line: "Suggested Improvements:"

    Example Output Structure:
    Contributing Factors:
    * Factor 1 (e.g., High Blood Pressure (Stage 1: 135/85 mmHg))
    * Factor 2 (e.g., BMI indicating Overweight (27.5))
    * Factor 3 (e.g., Smoking Reported)

    Suggested Improvements:
    * Suggestion 1 (e.g., Consider discussing blood pressure management options with a doctor.)
    * Suggestion 2 (e.g., Focus on achieving or maintaining a healthy weight through balanced diet and regular exercise.)
    * Suggestion 3 (e.g., Explore resources for smoking cessation.)
    * Suggestion 4 (e.g., Regular check-ups with a healthcare provider are recommended.)
    """

    try:
        print("Sending prompt to Gemini API...")
        response = model.generate_content(prompt)
        response_text = response.text
        print("Received response from Gemini API.")
        # print("--- Gemini Response ---")
        # print(response_text)
        # print("-----------------------")


        # --- Parse the response ---
        factors = []
        suggestions = []
        current_list = None

        lines = response_text.strip().split('\n')
        for line in lines:
            line = line.strip()
            if line.lower().startswith("contributing factors:"):
                current_list = factors
                # Remove the header line itself if needed, or handle lines starting with '*'/'•'/'-'
                line_content = line.split(":", 1)[1].strip()
                if line_content and not line_content.startswith(("*", "-", "•")):
                    factors.append(line_content)
                continue # Move to next line
            elif line.lower().startswith("suggested improvements:"):
                current_list = suggestions
                 # Remove the header line itself if needed
                line_content = line.split(":", 1)[1].strip()
                if line_content and not line_content.startswith(("*", "-", "•")):
                    suggestions.append(line_content)
                continue # Move to next line

            # Add lines starting with list markers to the current list
            if current_list is not None and line.startswith(("* ", "- ", "• ")):
                current_list.append(line[2:].strip()) # Add content after the marker
            elif current_list is not None and line: # Add non-empty lines if parser missed marker
                current_list.append(line)


        # Basic fallback if parsing fails to populate lists
        if not factors:
             factors.append("Could not automatically parse factors from AI response.")
        if not suggestions:
             suggestions.append("Please review the health data and consult a healthcare provider.")

        return factors, suggestions

    except Exception as e:
        print(f"Error calling Gemini API or parsing response: {e}")
        traceback.print_exc()
        return ["Error generating AI suggestions."], ["Could not connect to the suggestion service. Please consult a healthcare provider."]


# --- Flask Routes ---

@app.route('/')
def home():
    """Renders the home page with the input form."""
    print("Serving home page (index.html)")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handles form submission, preprocesses data, predicts, and shows results."""
    print("\nReceived prediction request (POST).")

    if pipeline is None or not expected_features:
        print("Error: Model pipeline not loaded properly.")
        flash("Error: The prediction model is not available.", "error")
        return redirect(url_for('home'))

    form_data = request.form.to_dict()
    print("Form Data Received:", form_data)

    try:
        # --- 1. Data Extraction and Type Conversion ---
        input_data = {}
        required_fields = {
            'age_years': int, 'height': float, 'weight': float, 'ap_hi': int, 'ap_lo': int,
            'gender': int, 'cholesterol': int, 'gluc': int, 'smoke': int, 'alco': int, 'active': int
        }
        for field, field_type in required_fields.items():
             value = form_data.get(field)
             if value is None or value == '': raise ValueError(f"Missing required field: {field}")
             try: input_data[field] = field_type(value)
             except (ValueError, TypeError): raise ValueError(f"Invalid input type for {field}")

        # --- 2. Feature Engineering (Replicate from Notebook) ---
        print("Engineering features (BMI, Pulse Pressure, BP Category)...")
        height_m = input_data['height'] / 100
        if height_m <= 0: raise ValueError("Height must be positive.")
        input_data['bmi'] = round(input_data['weight'] / (height_m ** 2), 2)
        input_data['pulse_pressure'] = input_data['ap_hi'] - input_data['ap_lo']
        input_data['bp_category'] = map_bp_to_category(input_data['ap_hi'], input_data['ap_lo'])
        print(f"  - Calculated BMI: {input_data['bmi']}, Pulse Pressure: {input_data['pulse_pressure']}, BP Category: {input_data['bp_category']}")

        # --- 3. Create DataFrame in Correct Order ---
        final_input_dict = {feature: input_data.get(feature) for feature in expected_features}
        if None in final_input_dict.values():
             missing = [k for k, v in final_input_dict.items() if v is None]
             raise ValueError(f"Internal error: Missing value for features: {missing}")
        input_df = pd.DataFrame([final_input_dict])[expected_features]
        print("\nInput DataFrame prepared for pipeline:\n", input_df.to_string())

        # --- 4. Make Prediction ---
        print("Making prediction using the loaded pipeline...")
        prediction_proba = pipeline.predict_proba(input_df)[0]
        probability_cvd = prediction_proba[1]
        risk_score = round(probability_cvd * 100, 2)
        print(f"Predicted probability of CVD: {probability_cvd:.4f} (Risk Score: {risk_score}%)")

        # --- 5. Get Suggestions from Gemini API ---
        print("Requesting suggestions from Gemini API...")
        # Pass the original input_data (with readable values if possible, but numerical needed too)
        # and the risk score to the Gemini function.
        factors, suggestions = get_gemini_suggestions(input_data, risk_score)
        print(f"  - Factors Received: {factors}")
        print(f"  - Suggestions Received: {suggestions}")


        # --- 6. Render Results Page ---
        print("Rendering results page...")
        return render_template('results.html',
                               risk_score=risk_score,
                               contributing_factors=factors, # Use Gemini results
                               suggested_improvements=suggestions # Use Gemini results
                               )

    except ValueError as ve:
        print(f"Data Validation Error: {ve}")
        flash(f"Input Error: {ve}. Please correct the input and try again.", "error")
        return redirect(url_for('home'))
    except Exception as e:
        print(f"ERROR during prediction process: {e}")
        print("Traceback:")
        traceback.print_exc()
        flash("An unexpected error occurred during risk assessment.", "error")
        return redirect(url_for('home'))

# --- Run the App ---
if __name__ == '__main__':
    # Set debug=False for production!
    app.run(debug=True, host='0.0.0.0', port=5000)