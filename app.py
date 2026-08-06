import streamlit as st # type: ignore
from PIL import Image # type: ignore
from ultralytics import YOLO # type: ignore

# YOLO Model Loader
@st.cache_resource
def load_yolo_model(model_name):
    model_map = {
        "YOLOv8m": "v8best.pt",
        "YOLOv8n": "v8nbest.pt",
        "YOLOv9c": "9cbest.pt"
    }
    return YOLO(model_map[model_name])

# Predict Function
def predict_surgical_tools(model, image):
    results = model.predict(image, imgsz=640, conf=0.4)
    return results[0]

# Custom CSS
def inject_custom_css():
    st.markdown("""
    <style>
       body, p, li, ul, ol, h1, h2, h3, h4, h5, h6, label, div {
            color: white !important;
        }
        body {
            background-color: #0E1117; 
        }
        h1, h2, h3 {
            color: #ffffff;
        }
        .stButton > button {
            background-color: #2c6df0;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 0.5rem 1rem;
            font-weight: 500;
        }
        .stButton > button:hover {
            background-color: #1e4cc9;
        }
        [data-testid="stSidebar"] {
            background-color: black;
        }
        .stImage > img {
            border-radius: 4px;
        }
        .section {
            margin-bottom: 2rem;
            color : white;
        }
    </style>
    """, unsafe_allow_html=True)

st.set_page_config(
    page_title="SURGI-YOLO",
    page_icon="🔬",
    layout="centered"
)
inject_custom_css()

# Sidebar Navigation
with st.sidebar:
    st.markdown("<h1 style='color: white;'>Navigation</h1>", unsafe_allow_html=True)
    page = st.radio("", ["Home", "Upload", "About"])

# Class Labels
CLASS_NAMES = {
    0: "Scalpel",
    1: "Straight Dissection Clamp",
    2: "Straight Mayo Scissor",
    3: "Curved Mayo Scissor"
}

# Home Page
if page == "Home":
    st.markdown("<h1 style='color: white;'>🔬 SURGI-YOLO</h1>", unsafe_allow_html=True)
    st.markdown("**Surgical Tool Detection System**")

    st.markdown("""
    <div class="section">
        <h2>Welcome</h2>
        <p>SURGI-YOLO is an AI-powered system for detecting surgical instruments in medical images using YOLO models.</p>
    </div>
    
    <div class="section">
        <h2>Features</h2>
        <ul>
            <li><b>Multiple Models:</b> YOLOv8, YOLOv9 support</li>
            <li><b>Real-time Detection:</b> Immediate bounding box results</li>
            <li><b>Detailed Output:</b> Tool labels and confidence scores</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Upload Camera Page
elif page == "Upload":
    st.markdown("<h1 style='color: white;'>Surgical Tool Detection</h1>", unsafe_allow_html=True)

    input_mode = st.radio("Choose input type:", ["Upload Image", "Use Camera"])
    model_option = st.selectbox("Select YOLO Model", ["YOLOv8m", "YOLOv8n", "YOLOv9c"], index=0)
    model = load_yolo_model(model_option)

    if input_mode == "Upload Image":
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])

        if uploaded_file:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width=True)

            if st.button("Analyze Image"):
                with st.spinner("Processing..."):
                    results = predict_surgical_tools(model, image)
                    st.image(results.plot(), use_container_width=True)

                    boxes = results.boxes
                    if boxes and len(boxes) > 0:
                        detected_tools = [CLASS_NAMES.get(int(box.cls[0]), f"Unknown ({int(box.cls[0])})") for box in boxes]

                        st.markdown("Detected Surgical Tools")
                        st.write(", ".join(sorted(set(detected_tools))))

                        st.table({
                            "Tool Name": detected_tools,
                            "Class ID": [int(box.cls[0]) for box in boxes],
                            "Confidence": [f"{float(box.conf[0]):.2f}" for box in boxes]
                        })
                    else:
                        st.info("No surgical tools detected.")

    elif input_mode == "Use Camera":
        stframe = st.empty()
        run = st.checkbox("Start Camera")

        if run:
            import cv2 # type: ignore
            cap = cv2.VideoCapture(1)
            st.info("Uncheck the checkbox to 'Stop Camera' or end live detection.")

            while run:
                ret, frame = cap.read()
                if not ret:
                    st.warning("Unable to access camera.")
                    break

                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = model.predict(frame_rgb, imgsz=640, conf=0.4)[0]
                plotted = results.plot()

                stframe.image(plotted, channels="RGB", use_container_width=True)

                boxes = results.boxes
                if boxes and len(boxes) > 0:
                    detected_tools = [CLASS_NAMES.get(int(box.cls[0]), f"Unknown ({int(box.cls[0])})") for box in boxes]

                    st.markdown("Detected Surgical Tools (Live)")
                    st.write(", ".join(sorted(set(detected_tools))))

                    st.table({
                        "Tool Name": detected_tools,
                        "Class ID": [int(box.cls[0]) for box in boxes],
                        "Confidence": [f"{float(box.conf[0]):.2f}" for box in boxes]
                    })
                else:
                    st.info("No surgical tools detected.")

            cap.release()

#  About Page
elif page == "About":
    st.markdown("<h1 style='color: white;'>About SURGI-YOLO</h1>", unsafe_allow_html=True)
    st.markdown("""
    <div class="section">
        <h2>Technology Stack</h2>
        <ul>
            <li><b>YOLOv8 / YOLOv9:</b> Object detection backbone</li>
            <li><b>Streamlit:</b> UI development framework</li>
            <li><b>PyTorch:</b> Deep learning runtime</li>
        </ul>
    </div>

    <div class="section">
        <h2>Use Cases</h2>
        <ul>
            <li>Surgical skill assessment</li>
            <li>Instrument tracking in real-time</li>
            <li>Research & training in robotic surgery</li>
        </ul>
    </div>

    <div class="section">
        <h2>Contact</h2>
        <p>📧 contact@surgiyolo.example</p>
    </div>
    """, unsafe_allow_html=True)
