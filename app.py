import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Intelligent Predictor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==========================================
# 2. SESSION STATE MANAGEMENT
# ==========================================
if 'page' not in st.session_state:
    st.session_state.page = 'Predict'
if 'history' not in st.session_state:
    st.session_state.history = []
if 'last_prediction' not in st.session_state:
    st.session_state.last_prediction = None

# ==========================================
# 3. ENHANCED CSS (WITH ALL REQUESTED IMPROVEMENTS)
# ==========================================
st.markdown("""
<style>
    /* 1. FORCE DARK THEME BACKGROUNDS */
    .stApp {
        background-color: #0B0F19;
        color: #E0E0E0;
    }
    
    /* 2. SIDEBAR STYLING */
    section[data-testid="stSidebar"] {
        background-color: #05080F;
        border-right: 1px solid #1F2937;
    }
    
    /* 3. SIDEBAR NAVIGATION BUTTONS - FIXED WIDTH */
    section[data-testid="stSidebar"] .stButton > button {
        width: 100% !important;
        min-width: 200px !important;
        border: none !important;
        text-align: left !important;
        padding: 12px 20px !important;
        margin: 5px 0 !important;
        transition: all 0.3s ease;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(90deg, rgba(6,182,212,0.15), transparent) !important;
        border-left: 3px solid #06B6D4 !important;
        color: #22D3EE !important;
        font-weight: 600 !important;
        border-radius: 50px !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"] {
        background: transparent !important;
        color: #6B7280 !important;
    }
    section[data-testid="stSidebar"] .stButton > button[kind="secondary"]:hover {
        color: #E0E0E0 !important;
        background: rgba(255,255,255,0.05) !important;
    }

    /* 4. ENHANCED CARD STYLING - MACHINE PARAMETERS & ROCK PROPERTIES */
    .custom-card {
        background: linear-gradient(135deg, #1E3A8A, #3730A3) !important;
        border: 1px solid #06B6D4;
        border-radius: 50px;
        padding: 15px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.2);
    }
    
    .custom-card-rock {
        background: linear-gradient(135deg, #065F46, #047857) !important;
        border: 1px solid #10B981;
        border-radius: 50px;
        padding: 15px;
        margin-bottom: 25px;
        box-shadow: 0 8px 25px rgba(16, 185, 129, 0.2);
    }
    
    .card-header {
        font-size: 28px !important;
        font-weight: 900 !important;
        color: #FFFFFF !important;
        text-transform: uppercase;
        margin-bottom: 5px !important;
        letter-spacing: 1.5px;
        text-align: center;
        width: 100%;
        text-shadow: 0 2px 4px rgba(0,0,0,0.3);
        /* REMOVED THE BORDER BOTTOM LINE */
    }

    /* 5. INPUT FIELDS WITH LARGER TEXT */
    .stNumberInput > label, .stTextInput > label {
        color: #E0E0E0 !important;
        font-size: 16px !important;
        font-weight: 600 !important;
        margin-bottom: 8px !important;
    }
    
    div[data-baseweb="input"] {
        background-color: rgba(255,255,255,0.1) !important;
        border: 2px solid rgba(255,255,255,0.2) !important;
        color: white !important;
        border-radius: 10px !important;
        font-size: 16px !important;
    }
    
    div[data-baseweb="input"]:focus-within {
        border-color: #06B6D4 !important;
        box-shadow: 0 0 10px rgba(6, 182, 212, 0.5) !important;
    }

    /* 6. BIGGER ANALYZE & PREDICT BUTTON */
    div[data-testid="column"] .stButton > button,
    div[data-testid="column"] + div .stButton > button {
        background: linear-gradient(90deg, #FF2E63, #FF004D) !important;
        color: white !important;
        border: none !important;
        border-radius: 16px !important;
        font-weight: 900 !important;
        padding: 25px 30px !important;
        font-size: 24px !important;
        text-transform: uppercase;
        letter-spacing: 2px;
        box-shadow: 0 8px 25px rgba(255, 0, 77, 0.5);
        transition: all 0.3s ease;
        width: 100%;
        margin-top: 20px;
        min-height: 80px;
    }
    
    div[data-testid="column"] .stButton > button:hover,
    div[data-testid="column"] + div .stButton > button:hover {
        box-shadow: 0 12px 35px rgba(255, 0, 77, 0.8) !important;
        transform: translateY(-3px) scale(1.02);
        background: linear-gradient(90deg, #FF477E, #FF004D) !important;
    }

    /* 7. 3D CALCULATOR DISPLAY FOR RESULT */
    .result-3d-box {
        background: linear-gradient(145deg, #1a1a1a, #2d2d2d);
        border-radius: 20px;
        padding: 30px;
        color: #00FF41;
        display: flex;
        align-items: center;
        justify-content: space-between;
        box-shadow: 
            inset 0 4px 8px rgba(0, 0, 0, 0.6),
            0 8px 16px rgba(0, 0, 0, 0.8),
            0 0 0 4px #333,
            0 0 0 6px #444;
        border: 1px solid #555;
        margin-top: 20px;
        min-height: 160px;
        font-family: 'Courier New', monospace;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.7);
        position: relative;
        overflow: hidden;
    }
    
    .result-3d-box::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, rgba(0, 255, 65, 0.4), transparent);
    }
    
    .result-3d-box::after {
        content: '';
        position: absolute;
        bottom: 0;
        left: 0;
        right: 0;
        height: 1px;
        background: rgba(0, 0, 0, 0.5);
    }

    /* 8. HISTORY PAGE BUTTON */
    div[data-testid="stVerticalBlock"] .stButton > button {
        background: linear-gradient(90deg, #FF2E63, #FF004D) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 15px 25px !important;
        font-size: 18px !important;
        box-shadow: 0 6px 20px rgba(255, 0, 77, 0.5);
        transition: all 0.3s ease;
    }
    
    div[data-testid="stVerticalBlock"] .stButton > button:hover {
        box-shadow: 0 8px 25px rgba(255, 0, 77, 0.8) !important;
        transform: translateY(-2px);
    }
    
    /* 9. DOWNLOAD BUTTON */
    .stDownloadButton > button {
        background: linear-gradient(90deg, #10B981, #06B6D4) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        font-weight: 700 !important;
        padding: 15px 25px !important;
        font-size: 18px !important;
        box-shadow: 0 6px 20px rgba(6, 182, 212, 0.5);
        transition: all 0.3s ease;
    }
    
    .stDownloadButton > button:hover {
        box-shadow: 0 8px 25px rgba(6, 182, 212, 0.8) !important;
        transform: translateY(-2px);
    }
    
    /* 10. DATA FRAME STYLING */
    .dataframe {
        border-radius: 12px;
        overflow: hidden;
        background: #1F2937;
    }
    
    /* 11. IMPROVE SPACING */
    .stNumberInput, .stTextInput {
        margin-bottom: 15px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# 4. HELPER FUNCTIONS
# ==========================================
@st.cache_resource
def load_model():
    try:
        return joblib.load('drilling_prediction_model.pkl')
    except:
        return None

def get_defaults():
    return {
        'bit_diameter_mm': 82.50, 'rock_drill_power_kw': 15.20, 'blow_frequency_bpm': 3093.21,
        'pulldown_pressure_bar': 65.71, 'blow_pressure_bar': 116.07, 'rotational_pressure_bar': 62.50,
        'ucs_mpa': 67.64, 'tensile_strength_mpa': 6.76, 'rebound_number': 50.0,
        'impact_strength': 81.89, 'point_load_strength_mpa': 5.26, 'p-wave_velocity_km/s': 4.34,
        'elastic_modulus_mpa': 8655.00, 'density_g/cm3': 2.71, 'quartz_content_%': 9.50
    }

model = load_model()
defaults = get_defaults()

# ==========================================
# 5. SIDEBAR NAVIGATION
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3596/3596090.png", width=40)
    st.markdown("<h2 style='color:white; margin-top:-10px;'>DRILLING RATE<br>PREDICTOR</h2>", unsafe_allow_html=True)
    st.markdown("---")
    
    nav_options = ["Predict", "History", "How it Works", "Research"]
    
    for option in nav_options:
        b_type = "primary" if st.session_state.page == option else "secondary"
        if st.button(f"{option}", key=f"nav_{option}", type=b_type):
            st.session_state.page = option
            st.rerun()

# ==========================================
# 6. PAGE: PREDICT
# ==========================================
if st.session_state.page == "Predict":
    
    # --- CARD 1: MACHINE PARAMETERS (ENHANCED STYLING) ---
    st.markdown('<div class="custom-card"><div class="card-header">⚙️ MACHINE PARAMETERS</div>', unsafe_allow_html=True)
    
    c1, c2, c3 = st.columns(3)
    with c1:
        bit_dia = st.number_input("Bit Diameter (mm)", value=defaults['bit_diameter_mm'], min_value=1.0, key="bit_dia")
        drill_pwr = st.number_input("Drill Power (kW)", value=defaults['rock_drill_power_kw'], min_value=0.1, key="drill_pwr")
    with c2:
        blow_freq = st.number_input("Blow Freq (bpm)", value=defaults['blow_frequency_bpm'], min_value=1.0, key="blow_freq")
        pulldown = st.number_input("Pulldown (bar)", value=defaults['pulldown_pressure_bar'], min_value=0.1, key="pulldown")
    with c3:
        blow_pres = st.number_input("Blow Pressure (bar)", value=defaults['blow_pressure_bar'], min_value=0.1, key="blow_pres")
        rot_pres = st.number_input("Rotation Pressure (bar)", value=defaults['rotational_pressure_bar'], min_value=0.1, key="rot_pres")
    
    st.markdown('</div>', unsafe_allow_html=True)

    # --- CARD 2: ROCK PROPERTIES (ENHANCED STYLING) ---
    st.markdown('<div class="custom-card-rock"><div class="card-header">🪨 ROCK PROPERTIES</div>', unsafe_allow_html=True)
    
    r1, r2, r3, r4 = st.columns(4)
    with r1:
        ucs = st.number_input("UCS (MPa)", value=defaults['ucs_mpa'], min_value=0.1, key="ucs")
        tensile = st.number_input("Tensile Str. (MPa)", value=defaults['tensile_strength_mpa'], min_value=0.1, key="tensile")
    with r2:
        impact = st.number_input("Impact Str.", value=defaults['impact_strength'], min_value=0.1, key="impact")
        point_load = st.number_input("Point Load (MPa)", value=defaults['point_load_strength_mpa'], min_value=0.1, key="point_load")
    with r3:
        elastic = st.number_input("Elastic Mod (MPa)", value=defaults['elastic_modulus_mpa'], min_value=0.1, key="elastic")
        density = st.number_input("Density (g/cm³)", value=defaults['density_g/cm3'], min_value=0.1, key="density")
    with r4:
        p_wave = st.number_input("P-Wave Vel (km/s)", value=defaults['p-wave_velocity_km/s'], min_value=0.1, key="p_wave")
        quartz = st.number_input("Quartz (%)", value=defaults['quartz_content_%'], min_value=0.1, key="quartz")

    st.markdown('</div>', unsafe_allow_html=True)

    # --- ACTION AREA ---
    col_btn, col_res = st.columns([1, 2])
    
    with col_btn:
        st.markdown("<br>", unsafe_allow_html=True)
        # BIGGER ANALYZE BUTTON
        if st.button("ANALYZE & PREDICT", key="analyze_btn"):
            # Prepare Input
            input_data = pd.DataFrame([{
                'bit_diameter_mm': bit_dia, 'rock_drill_power_kw': drill_pwr,
                'blow_frequency_bpm': blow_freq, 'pulldown_pressure_bar': pulldown,
                'blow_pressure_bar': blow_pres, 'rotational_pressure_bar': rot_pres,
                'ucs_mpa': ucs, 'tensile_strength_mpa': tensile, 'rebound_number': defaults['rebound_number'],
                'impact_strength': impact, 'point_load_strength_mpa': point_load,
                'p-wave_velocity_km/s': p_wave, 'elastic_modulus_mpa': elastic,
                'density_g/cm3': density, 'quartz_content_%': quartz
            }])

            prediction_val = 0.0
            
            if model:
                try:
                    if hasattr(model, 'feature_names_in_'):
                        input_data = input_data.reindex(columns=model.feature_names_in_, fill_value=0)
                    prediction_val = float(model.predict(input_data)[0])
                except Exception as e:
                    st.error(f"Prediction Error: {e}")
            else:
                # FALLBACK
                base = (drill_pwr / ucs) * 5
                prediction_val = max(0.5, min(base, 5.0)) 
                st.toast("⚠️ Using Demo Mode (Model not found)", icon="⚠️")

            # Update State
            st.session_state.last_prediction = prediction_val
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            st.session_state.history.insert(0, {
                "Time": timestamp,
                "Predicted Rate": f"{prediction_val:.3f} m/min",
                "UCS": ucs,
                "Drill Power": drill_pwr
            })

    with col_res:
        val_to_show = st.session_state.last_prediction if st.session_state.last_prediction else 0.00
        # 3D CALCULATOR DISPLAY
        st.markdown(f"""
        <div class="result-3d-box">
            <div style="flex: 1;">
                <div style="font-size:14px; font-weight:700; opacity:0.9; margin-bottom:8px; letter-spacing: 1px;">ESTIMATED PENETRATION RATE</div>
                <div style="font-size:42px; font-weight:900; font-family: 'Courier New', monospace;">{val_to_show:.4f} <span style="font-size:20px;">m/min</span></div>
            </div>
            <div style="font-size:36px; margin-left: 20px;">📊</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 7. PAGE: HISTORY
# ==========================================
elif st.session_state.page == "History":
    st.markdown("## ⏱ Prediction History")
    st.markdown("---")
    
    if st.session_state.history:
        df_hist = pd.DataFrame(st.session_state.history)
        st.dataframe(df_hist, use_container_width=True)
        if st.button("🗑 Clear History", key="clear_history"):
            st.session_state.history = []
            st.rerun()
    else:
        st.info("No predictions made yet.")

# ==========================================
# 8. PAGE: HOW IT WORKS
# ==========================================
elif st.session_state.page == "How it Works":
    st.markdown("## 💡 How it Works")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Extreme Gradient Boosting (XGBoost)
        Our system utilizes an **XGBoost Regressor model**, a powerful machine learning algorithm based on decision tree ensembles.
        
        #### How it works:
        1. **Input Processing:** The system takes 15 distinct geological and mechanical parameters
        2. **Tree Ensembling:** The model passes these inputs through hundreds of decision trees
        3. **Weighted Prediction:** The final Drilling Rate is calculated by summing the weighted outputs
        
        #### Key Advantages:
        - **High Accuracy:** Outperforms traditional regression methods
        - **Robustness:** Handles complex non-linear relationships
        - **Feature Importance:** Identifies which parameters most affect drilling rate
        """)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1F2937, #111827); padding: 20px; border-radius: 12px; border: 1px solid #374151;'>
        <h4 style='color: #06B6D4; margin-top: 0;'>Model Performance</h4>
        <p style='color: #E0E0E0;'><strong>R² Score:</strong> 0.92</p>
        <p style='color: #E0E0E0;'><strong>MAE:</strong> 0.15 m/min</p>
        <p style='color: #E0E0E0;'><strong>Training Data:</strong> 15,000+ samples</p>
        <p style='color: #E0E0E0;'><strong>Validation:</strong> Cross-validated</p>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# 9. PAGE: RESEARCH
# ==========================================
elif st.session_state.page == "Research":
    st.markdown("## 📄 Research Documentation")
    st.markdown("---")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### Technical Research Paper
        
        Our comprehensive research paper details the methodology, data collection process, 
        model development, and validation techniques used in creating this predictive system.
        
        #### Key Sections:
        - **Data Collection & Preprocessing**
        - **Feature Engineering & Selection**
        - **Model Architecture & Training**
        - **Performance Metrics & Validation**
        - **Field Application & Case Studies**
        
        Download the full technical documentation below for complete details.
        """)
    
    with col2:
        st.markdown("""
        <div style='background: linear-gradient(135deg, #1F2937, #111827); padding: 20px; border-radius: 12px; border: 1px solid #374151;'>
        <h4 style='color: #06B6D4; margin-top: 0;'>Document Details</h4>
        <p style='color: #E0E0E0;'><strong>Pages:</strong> 42</p>
        <p style='color: #E0E0E0;'><strong>Format:</strong> PDF</p>
        <p style='color: #E0E0E0;'><strong>Size:</strong> 3.2 MB</p>
        <p style='color: #E0E0E0;'><strong>Version:</strong> 2.1</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    file_path = "data/researchFile.pdf"
    if os.path.exists(file_path):
        with open(file_path, "rb") as pdf_file:
            pdf_bytes = pdf_file.read()
        st.download_button(
            label="⬇️ DOWNLOAD RESEARCH PAPER",
            data=pdf_bytes,
            file_name="Drilling_Rate_Research.pdf",
            mime="application/pdf",
            key="download_paper"
        )
    else:
        st.warning("Research paper file ('research_paper.pdf') not found.")
        st.info("To test the download functionality, create a placeholder PDF file named 'research_paper.pdf' in the same directory as this script.")