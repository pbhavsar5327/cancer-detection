import os
import numpy as np
import cv2
import tensorflow as tf
from flask import Flask, request, render_template, jsonify
from tensorflow.keras.models import load_model
from PIL import Image
import base64
import google.generativeai as genai
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

# ─── LOAD ENV ─────────────────────────────────────────────
load_dotenv()
GEMINI_KEY = os.getenv('GEMINI_API_KEY')

# ─── CONFIGURE GEMINI ─────────────────────────────────────
if GEMINI_KEY:
    genai.configure(api_key=GEMINI_KEY)
    # gemini-1.5-flash supports Google Search grounding
    gemini_model = genai.GenerativeModel('gemini-1.5-flash')
    print("Gemini AI connected!")
else:
    gemini_model = None
    print("Warning: GEMINI_API_KEY not found in .env file!")

app = Flask(__name__)

# ─── CONFIG ───────────────────────────────────────────────
MODEL_PATH = 'model/cancer_model.h5'
UPLOAD_DIR = 'static/uploads'
IMG_SIZE   = 224

if not os.path.exists(UPLOAD_DIR):
    os.makedirs(UPLOAD_DIR)

# ─── LOAD ML MODEL ────────────────────────────────────────
print("Loading ML model...")
model = load_model(MODEL_PATH)
print("ML Model loaded successfully!")

# ─── GRAD-CAM ─────────────────────────────────────────────
def generate_gradcam(img_array):
    try:
        last_conv_layer = None
        for layer in reversed(model.layers):
            if isinstance(layer, tf.keras.layers.Conv2D):
                last_conv_layer = layer.name
                break
        if not last_conv_layer:
            return None

        grad_model = tf.keras.models.Model(
            inputs=model.inputs,
            outputs=[model.get_layer(last_conv_layer).output, model.output]
        )

        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img_array)
            loss = predictions[:, 0]

        grads        = tape.gradient(loss, conv_outputs)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        conv_outputs = conv_outputs[0]
        heatmap      = conv_outputs @ pooled_grads[..., tf.newaxis]
        heatmap      = tf.squeeze(heatmap)
        heatmap      = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-8)
        return heatmap.numpy()
    except Exception as e:
        print(f"Grad-CAM error: {e}")
        return None

def overlay_gradcam(img_path, heatmap):
    try:
        img             = cv2.imread(img_path)
        img             = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))
        heatmap_uint8   = np.uint8(255 * heatmap_resized)
        heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        superimposed    = cv2.addWeighted(img, 0.55, heatmap_colored, 0.45, 0)
        _, buffer       = cv2.imencode('.jpg', superimposed)
        return "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
    except Exception as e:
        print(f"Overlay error: {e}")
        return None

# ─── CANCER STAGE LOGIC ───────────────────────────────────
# cancer=0, normal=1
# LOW score = cancer, HIGH score = normal
# 5 stages: Stage 0 (precancerous), Stage 1, 2, 3, 4
def get_cancer_details(score):
    pct = score * 100

    if pct >= 50:
        # Normal — no cancer
        return {
            "result":        "Normal / No Cancer Detected",
            "stage":         "None",
            "stage_number":  0,
            "curable":       "N/A",
            "action":        "No immediate action needed. Maintain a healthy lifestyle and schedule regular checkups every 6 months.",
            "recovery":      "100%",
            "survival_rate": "100%",
            "color":         "green",
            "risk":          "Low Risk",
            "is_cancer":     False,
            "symptoms":      "No abnormal symptoms detected.",
            "treatments":    "N/A"
        }
    elif pct >= 42:
        # Stage 0 — precancerous / in-situ
        return {
            "result":        "Precancerous Cells Detected",
            "stage":         "Stage 0 — Pre-Cancer (In Situ)",
            "stage_number":  0,
            "curable":       "Almost Always Curable",
            "action":        "Schedule an immediate biopsy and specialist consultation. Stage 0 means abnormal cells are present but have not spread. Early intervention is key.",
            "recovery":      "99%",
            "survival_rate": "99%",
            "color":         "orange",
            "risk":          "Very Low Risk",
            "is_cancer":     True,
            "symptoms":      "Usually no symptoms. Detected only through screening.",
            "treatments":    "Surgery (lumpectomy), radiation therapy, or close monitoring depending on location."
        }
    elif pct >= 32:
        # Stage 1 — localized
        return {
            "result":        "Cancer Detected",
            "stage":         "Stage 1 — Early Stage (Localized)",
            "stage_number":  1,
            "curable":       "Highly Curable",
            "action":        "Consult an oncologist immediately. Cancer is small and localized. Surgery or targeted therapy at this stage gives 85-95% survival rate. Do not delay.",
            "recovery":      "90%",
            "survival_rate": "85% - 95%",
            "color":         "orange",
            "risk":          "Low-Medium Risk",
            "is_cancer":     True,
            "symptoms":      "Mild persistent cough, slight chest discomfort, minor fatigue.",
            "treatments":    "Surgery, radiation therapy, targeted therapy, immunotherapy."
        }
    elif pct >= 22:
        # Stage 2 — regional
        return {
            "result":        "Cancer Detected",
            "stage":         "Stage 2 — Moderate Stage (Regional)",
            "stage_number":  2,
            "curable":       "Curable with Treatment",
            "action":        "Seek oncologist consultation immediately. Cancer has grown but is still manageable. Combined treatment approach needed. Time is critical.",
            "recovery":      "70%",
            "survival_rate": "60% - 75%",
            "color":         "orange",
            "risk":          "Medium Risk",
            "is_cancer":     True,
            "symptoms":      "Persistent cough, chest pain, shortness of breath, unexplained weight loss.",
            "treatments":    "Surgery + chemotherapy, radiation therapy, targeted therapy, immunotherapy."
        }
    elif pct >= 12:
        # Stage 3 — advanced regional
        return {
            "result":        "Cancer Detected",
            "stage":         "Stage 3 — Advanced Stage (Extensive Regional)",
            "stage_number":  3,
            "curable":       "Possibly Curable",
            "action":        "Urgent oncology consultation required. Cancer has spread to nearby lymph nodes. Aggressive combined treatment is needed immediately. Seek second opinion.",
            "recovery":      "40%",
            "survival_rate": "25% - 45%",
            "color":         "red",
            "risk":          "High Risk",
            "is_cancer":     True,
            "symptoms":      "Severe cough, blood in sputum, significant weight loss, bone pain, hoarseness.",
            "treatments":    "Chemotherapy + radiation (concurrent), immunotherapy, targeted therapy, clinical trials."
        }
    else:
        # Stage 4 — metastatic
        return {
            "result":        "Cancer Detected",
            "stage":         "Stage 4 — Critical Stage (Metastatic)",
            "stage_number":  4,
            "curable":       "Very Difficult — Palliative Care Recommended",
            "action":        "Immediate specialist consultation required. Cancer has spread to other organs. Focus on quality of life, pain management, and palliative care. Consult multiple specialists.",
            "recovery":      "15%",
            "survival_rate": "5% - 15%",
            "color":         "darkred",
            "risk":          "Critical Risk",
            "is_cancer":     True,
            "symptoms":      "Severe pain, extreme fatigue, difficulty breathing, jaundice, neurological symptoms.",
            "treatments":    "Palliative chemotherapy, immunotherapy, targeted therapy, pain management, hospice care."
        }

# ─── GEMINI ANALYSIS WITH INTERNET SEARCH ─────────────────
def get_gemini_analysis(details, confidence):
    if not gemini_model:
        return None
    try:
        stage         = details['stage']
        result        = details['result']
        survival_rate = details['survival_rate']
        treatments    = details['treatments']
        symptoms      = details['symptoms']

        prompt = f"""You are a professional medical AI assistant with access to the latest medical research and internet information.

A patient's chest X-ray has been analyzed with these findings:
- Result: {result}
- Stage: {stage}
- Survival Rate: {survival_rate}
- Current Symptoms: {symptoms}
- Standard Treatments: {treatments}
- Confidence Score: {confidence:.1f}%

Using the latest medical knowledge and research, please provide a comprehensive report with these EXACT sections:

## 📋 What This Diagnosis Means
Explain in simple, clear language what this stage means for the patient.

## 🏥 Latest Treatment Options (2024)
List the most current and effective treatment options available today including any new immunotherapy or targeted therapy breakthroughs.

## 💊 Medicines & Therapies
Mention specific medicines, drugs, or therapies commonly used for this stage. Include both conventional and newer options.

## 🥗 Lifestyle Changes to Make Immediately
List specific diet, exercise, sleep, and stress management changes the patient should make right now.

## ⚠️ Warning Signs to Watch For
List specific symptoms that mean the patient needs emergency care immediately.

## 💪 Recovery & Survival Statistics
Give realistic but encouraging statistics about survival and recovery for this stage based on latest research.

## 🔗 Recommended Next Steps
Give a clear step-by-step action plan for what the patient should do in the next 24 hours, 1 week, and 1 month.

Be factual, empathetic, and use simple language. Base your answer on the latest medical guidelines from WHO, American Cancer Society, and recent research."""

        response = gemini_model.generate_content(prompt)
        return response.text

    except Exception as e:
        print(f"Gemini error: {e}")
        return None

# ─── PREDICT ──────────────────────────────────────────────
def predict_image(img_path):
    img       = Image.open(img_path).convert('RGB')
    img       = img.resize((IMG_SIZE, IMG_SIZE))
    img_array = np.array(img).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array, verbose=0)
    confidence = float(prediction[0][0])
    details    = get_cancer_details(confidence)

    # Grad-CAM only if cancer detected
    gradcam_img = None
    if details['is_cancer']:
        heatmap = generate_gradcam(img_array)
        if heatmap is not None:
            gradcam_img = overlay_gradcam(img_path, heatmap)

    # Gemini AI analysis with internet info
    ai_analysis = get_gemini_analysis(details, confidence * 100)

    return confidence, details, gradcam_img, ai_analysis

# ─── ROUTES ───────────────────────────────────────────────
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})

    filepath = os.path.join(UPLOAD_DIR, file.filename)
    file.save(filepath)

    confidence, details, gradcam_img, ai_analysis = predict_image(filepath)

    # Remove internal fields from response
    details_copy = {k: v for k, v in details.items() if k not in ['is_cancer', 'stage_number']}

    return jsonify({
        'confidence':  round(confidence * 100, 2),
        'gradcam':     gradcam_img,
        'ai_analysis': ai_analysis,
        **details_copy
    })

@app.route('/chat', methods=['POST'])
def chat():
    if not gemini_model:
        return jsonify({'reply': 'Gemini AI is not configured. Please check your API key.'})

    data    = request.get_json()
    message = data.get('message', '')
    context = data.get('context', {})

    try:
        prompt = f"""You are a medical AI assistant with knowledge of the latest cancer research and treatments.

The patient's X-ray analysis showed:
- Result: {context.get('result', 'Unknown')}
- Stage: {context.get('stage', 'Unknown')}
- Risk: {context.get('risk', 'Unknown')}
- Recovery Chance: {context.get('recovery', 'Unknown')}
- Survival Rate: {context.get('survival_rate', 'Unknown')}
- Treatments: {context.get('treatments', 'Unknown')}

The patient is asking: "{message}"

Answer using the latest medical knowledge. Be clear, compassionate, factual, and use simple language.
Keep response under 200 words. If the question is about medicines or treatments, mention specific options."""

        response = gemini_model.generate_content(prompt)
        return jsonify({'reply': response.text})
    except Exception as e:
        return jsonify({'reply': f'Error getting response: {str(e)}'})

# ─── RUN ──────────────────────────────────────────────────
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)