import streamlit as st
import pandas as pd
import joblib
from fpdf import FPDF
from io import BytesIO

# Load model
model = joblib.load('ckd_model.pkl')

st.set_page_config(page_title="CKD Predictor", layout="centered")
st.markdown("""
    <style>
        /* Change + and - buttons color in number input fields */
        button[kind="icon"] {
            background-color: #0056b3 !important;  /* Dark Blue */
            color: black ;
            border-radius: 5px !important;
        }
        
        button[kind="icon"]:hover {
            background-color: #002855 !important;  /* Even Darker Blue */
        }

        /* Change "Download PDF" button color */
        div.stDownloadButton > button {
            background-color: #28a745 !important;  /* Green */
            color: white !important;
            font-size: 16px !important;
            padding: 10px 20px !important;
            border-radius: 8px !important;
            border: none !important;
        }

        div.stDownloadButton > button:hover {
            background-color: #218838 !important;  /* Darker Green */
        }
    </style>
""", unsafe_allow_html=True)


# Inject custom CSS
with open("style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
# Session page routing
if "page" not in st.session_state:
    st.session_state.page = "home"

# PDF generation
def generate_pdf_report(data_dict, prediction, probability):
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    pdf.cell(200, 10, txt="Chronic Kidney Disease Prediction Report", ln=1, align="C")
    pdf.ln(10)

    for key, val in data_dict.items():
        pdf.cell(200, 10, txt=f"{key}: {val}", ln=1)

    pdf.ln(10)
    pdf.cell(200, 10, txt=f"Prediction: {'CKD Detected' if prediction == 1 else 'No CKD Detected'}", ln=1)
    pdf.cell(200, 10, txt=f"CKD Probability: {probability * 100:.2f}%", ln=1)

    pdf_bytes = pdf.output(dest='S').encode('latin-1')
    return BytesIO(pdf_bytes)

# HOME PAGE
if st.session_state.page == "home":
    st.markdown("""
        <div class="container">
            <h1>Chronic Kidney Disease Prediction</h1>
            <p>🩺 What is Chronic Kidney Disease (CKD)?
Chronic Kidney Disease (CKD) is a serious medical condition where the kidneys slowly lose their ability to filter waste and excess fluids from the blood. It often develops silently over time and shows little or no symptoms in the early stages, making it hard to detect without medical tests. If left untreated, CKD can lead to kidney failure, requiring dialysis or a kidney transplant. It is commonly caused by diabetes, high blood pressure, and other underlying health issues.

CKD is a global health concern affecting millions, and early detection is crucial for managing the disease and preventing severe complications.

❓ Would you like to check your risk of CKD?
👉 Click the button below to start the prediction and get your personalized health report.</p>
        </div>
    """, unsafe_allow_html=True)

    if st.button("Start Prediction"):
        st.session_state.page = "predict"

# PREDICTION PAGE
elif st.session_state.page == "predict":
    st.markdown("<div class='container'>", unsafe_allow_html=True)
    st.header("Basic Information")

    # Inputs
    age = st.number_input("Age", min_value=0)
    bp = st.number_input("Blood Pressure", min_value=0)
    sg = st.number_input("Specific Gravity", min_value=0.0, format="%.3f")
    al = st.number_input("Albumin", min_value=0)
    su = st.number_input("Sugar", min_value=0)

    st.header("Lab Test Results")
    rbc = st.selectbox("Red Blood Cells", ["normal", "abnormal"])
    pc = st.selectbox("Pus Cell", ["normal", "abnormal"])
    pcc = st.selectbox("Pus Cell Clumps", ["present", "notpresent"])
    ba = st.selectbox("Bacteria", ["present", "notpresent"])
    hemo = st.number_input("Hemoglobin", min_value=0.0)
    dm = st.selectbox("Diabetes Mellitus", ['yes', 'no'])
    cad = st.selectbox("Coronary Artery Disease", ['yes', 'no'])
    sod = st.number_input("Sodium", min_value=0.0)
    pot = st.number_input("Potassium", min_value=0.0)
    wc = st.number_input("White Blood Cell Count", min_value=0)
    rc = st.number_input("Red Blood Cell Count", min_value=0.0)
    appet = st.selectbox("Appetite", ['good', 'poor'])
    pcv = st.number_input("Packed Cell Volume", min_value=0)
    ane = st.selectbox("Anemia", ['yes', 'no'])
    htn = st.selectbox("Hypertension", ['yes', 'no'])
    bgr = st.number_input("Blood Glucose Random", min_value=0)
    pe = st.selectbox("Pedal Edema", ['yes', 'no'])
    sc = st.number_input("Serum Creatinine", min_value=0.0)
    bu = st.number_input("Blood Urea", min_value=0.0)

    # Validation function
    def validate_input():
        if age < 18 or age > 120:
            st.error("Age must be between 18 and 120.")
            return False
        if bp < 30 or bp > 250:
            st.error("Blood Pressure must be between 30 and 250.")
            return False
        if sg < 1.005 or sg > 1.030:
            st.error("Specific Gravity must be between 1.005 and 1.030.")
            return False
        if al < 0 or al > 5:
            st.error("Albumin must be between 0 and 5.")
            return False
        if su < 0 or su > 5:
            st.error("Sugar must be between 0 and 5.")
            return False
        if rbc == "abnormal":
            st.error("Red Blood Cells should be normal for prediction.")
            return False
        if pc == "abnormal":
            st.error("Pus Cells should be normal for prediction.")
            return False
        if pcc != "notpresent":
            st.error("Pus Cell Clumps should be not present.")
            return False
        if ba != "notpresent":
            st.error("Bacteria should be not present.")
            return False
        if hemo < 3 or hemo > 20:
            st.error("Hemoglobin must be between 3 and 20 g/dL.")
            return False
        if dm not in ["yes", "no"]:
            st.error("Select a valid option for Diabetes Mellitus.")
            return False
        if cad not in ["yes", "no"]:
            st.error("Select a valid option for Coronary Artery Disease.")
            return False
        if sod < 120 or sod > 160:
            st.error("Sodium must be between 120 and 160 mEq/L.")
            return False
        if pot < 2.5 or pot > 6.5:
            st.error("Potassium must be between 2.5 and 6.5 mEq/L.")
            return False
        if wc < 4000 or wc > 11000:
            st.error("White Blood Cell Count must be between 4000 and 11000 /μL.")
            return False
        if rc < 3.00 or rc > 6.00:
            st.error("Red Blood Cell Count must be between 3 and 6 million cells/μL.")
            return False
        if appet not in ["good", "poor"]:
            st.error("Select a valid option for Appetite.")
            return False
        if pcv < 20 or pcv > 55:
            st.error("Packed Cell Volume must be between 20% and 55%.")
            return False
        if ane not in ["yes", "no"]:
            st.error("Select a valid option for Anemia.")
            return False
        if htn not in ["yes", "no"]:
            st.error("Select a valid option for Hypertension.")
            return False
        if bgr < 70 or bgr > 300:
            st.error("Blood Glucose Random must be between 70 and 300 mg/dL.")
            return False
        if pe not in ["yes", "no"]:
            st.error("Select a valid option for Pedal Edema.")
            return False
        if sc < 0.4 or sc > 15:
            st.error("Serum Creatinine must be between 0.4 and 15 mg/dL.")
            return False
        if bu < 5 or bu > 150:
            st.error("Blood Urea must be between 5 and 150 mg/dL.")
            return False
        return True

    # Prepare input for model
    input_data = pd.DataFrame([{
        'age': age, 'bp': bp, 'sg': sg, 'al': al, 'su': su,
        'rbc': rbc, 'pc': pc, 'pcc': pcc, 'ba': ba, 'hemo': hemo,
        'dm': dm, 'cad': cad, 'sod': sod, 'pot': pot, 'wc': wc,
        'rc': rc, 'appet': appet, 'pcv': pcv, 'ane': ane, 'htn': htn,
        'bgr': bgr, 'pe': pe, 'sc': sc, 'bu': bu
    }])

    if st.button("🔍 Predict CKD") and validate_input():
        prediction = model.predict(input_data)[0]
        proba_ckd = model.predict_proba(input_data)[0][1]

        if prediction == 1:
            st.error("🟥 Prediction: CKD Detected")
        else:
            st.success(f"🟩 Prediction: No CKD Detected\n🧪 CKD Risk Probability: {proba_ckd * 100:.2f}%")

        report_pdf = generate_pdf_report(input_data.iloc[0].to_dict(), prediction, proba_ckd)
        st.download_button(
            label="📥 Download PDF Report",
            data=report_pdf,
            file_name="ckd_report.pdf",
            mime='application/pdf'
        )

    if st.button("⬅ Back to Home"):
        st.session_state.page = "home"

    st.markdown("</div>", unsafe_allow_html=True)

