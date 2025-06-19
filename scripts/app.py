import streamlit as st
import torch
from torchvision import transforms, models
from PIL import Image
import pandas as pd
import os
import random
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import io
import requests
import cv2
import numpy as np
import xml.etree.ElementTree as ET
# ------------ ENHANCED DARK MODE TOGGLER ---------------- 
import streamlit as st

if "dark_mode" not in st.session_state: 
    st.session_state.dark_mode = False 
 
def toggle_mode(): 
    st.session_state.dark_mode = not st.session_state.dark_mode 

# Custom CSS for professional styling
def load_css():
    if st.session_state.dark_mode:
        st.markdown(
            """
            <style>
            /* Dark Mode Styling */
            .stApp {
                background: linear-gradient(135deg, #0f1419 0%, #1a1f2e 100%);
            }
            
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1200px;
            }
            
            /* Sidebar Styling */
            .css-1d391kg, [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #1e2329 0%, #2d3748 100%);
                border-right: 2px solid #4a5568;
            }
            
            .css-1d391kg .css-1v0mbdj {
                color: #e2e8f0;
            }
            
            /* Enhanced Button Styling */
            .stButton > button {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                font-size: 0.95rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(102, 126, 234, 0.3);
                width: 100%;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
                background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
            }
            
            /* Toggle Button Special Styling */
            .mode-toggle {
                background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%) !important;
                color: #2d3748 !important;
                border-radius: 25px !important;
                padding: 0.5rem 1.2rem !important;
                font-weight: 700 !important;
                margin-bottom: 1rem !important;
            }
            
            /* Selectbox Styling */
            .stSelectbox > div > div {
                background: #2d3748;
                border: 2px solid #4a5568;
                border-radius: 12px;
                color: #e2e8f0;
            }
            
            /* Card-like containers */
            .feature-card {
                background: rgba(45, 55, 72, 0.8);
                border: 1px solid #4a5568;
                border-radius: 16px;
                padding: 1.5rem;
                margin: 1rem 0;
                backdrop-filter: blur(10px);
                box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
            }
            
            /* Enhanced Typography */
            .dashboard-title {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.5rem;
                font-weight: 800;
                text-align: center;
                margin-bottom: 0.5rem;
            }
            
            .subtitle {
                color: #a0aec0;
                text-align: center;
                font-size: 1.1rem;
                margin-bottom: 2rem;
            }
            
            /* Navigation styling */
            .nav-section {
                background: rgba(45, 55, 72, 0.6);
                border-radius: 12px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid #667eea;
            }
            
            /* Expander styling */
            .streamlit-expanderHeader {
                background: rgba(45, 55, 72, 0.8);
                border-radius: 8px;
                border: 1px solid #4a5568;
            }
            
            /* Text and content styling */
            .stMarkdown, .stText {
                color: #e2e8f0;
            }
            
            /* Data display styling */
            .stDataFrame, .stTable {
                background: rgba(45, 55, 72, 0.8);
                border-radius: 12px;
                border: 1px solid #4a5568;
            }
            
            /* Loading and progress bars */
            .stProgress > div > div {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            
            /* Metrics styling */
            .metric-container {
                background: rgba(45, 55, 72, 0.8);
                border: 1px solid #4a5568;
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
            }
            </style>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <style>
            /* Light Mode Styling */
            .stApp {
                background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            }
            
            .main .block-container {
                padding-top: 2rem;
                padding-bottom: 3rem;
                max-width: 1200px;
                background: rgba(255, 255, 255, 0.9);
                border-radius: 20px;
                box-shadow: 0 10px 40px rgba(0, 0, 0, 0.1);
                margin-top: 1rem;
            }
            
            /* Sidebar Styling */
            .css-1d391kg, [data-testid="stSidebar"] {
                background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
                border-right: 2px solid #e2e8f0;
                box-shadow: 2px 0 10px rgba(0, 0, 0, 0.05);
            }
            
            /* Enhanced Button Styling */
            .stButton > button {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                font-size: 0.95rem;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(79, 172, 254, 0.3);
                width: 100%;
            }
            
            .stButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(79, 172, 254, 0.4);
                background: linear-gradient(135deg, #00f2fe 0%, #4facfe 100%);
            }
            
            /* Toggle Button Special Styling */
            .mode-toggle {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
                color: white !important;
                border-radius: 25px !important;
                padding: 0.5rem 1.2rem !important;
                font-weight: 700 !important;
                margin-bottom: 1rem !important;
            }
            
            /* Download Button Styling */
            .stDownloadButton > button {
                background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
                color: white;
                border: none;
                border-radius: 12px;
                padding: 0.75rem 1.5rem;
                font-weight: 600;
                transition: all 0.3s ease;
                box-shadow: 0 4px 15px rgba(17, 153, 142, 0.3);
            }
            
            .stDownloadButton > button:hover {
                transform: translateY(-2px);
                box-shadow: 0 6px 20px rgba(17, 153, 142, 0.4);
            }
            
            /* Selectbox Styling */
            .stSelectbox > div > div {
                background: white;
                border: 2px solid #e2e8f0;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0, 0, 0, 0.05);
            }
            
            /* Card-like containers */
            .feature-card {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #e2e8f0;
                border-radius: 16px;
                padding: 1.5rem;
                margin: 1rem 0;
                box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
                backdrop-filter: blur(10px);
            }
            
            /* Enhanced Typography */
            .dashboard-title {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.5rem;
                font-weight: 800;
                text-align: center;
                margin-bottom: 0.5rem;
            }
            
            .subtitle {
                color: #64748b;
                text-align: center;
                font-size: 1.1rem;
                margin-bottom: 2rem;
            }
            
            /* Navigation styling */
            .nav-section {
                background: rgba(255, 255, 255, 0.8);
                border-radius: 12px;
                padding: 1rem;
                margin: 0.5rem 0;
                border-left: 4px solid #4facfe;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            }
            
            /* Text input styling */
            .stTextArea textarea, .stTextInput input {
                border-radius: 12px;
                border: 2px solid #e2e8f0;
                background: rgba(255, 255, 255, 0.9);
                transition: all 0.3s ease;
            }
            
            .stTextArea textarea:focus, .stTextInput input:focus {
                border-color: #4facfe;
                box-shadow: 0 0 0 3px rgba(79, 172, 254, 0.1);
            }
            
            /* Expander styling */
            .streamlit-expanderHeader {
                background: rgba(255, 255, 255, 0.9);
                border-radius: 8px;
                border: 1px solid #e2e8f0;
            }
            
            /* Progress bars */
            .stProgress > div > div {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
            }
            
            /* Metrics styling */
            .metric-container {
                background: rgba(255, 255, 255, 0.9);
                border: 1px solid #e2e8f0;
                border-radius: 12px;
                padding: 1rem;
                text-align: center;
                box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
            }
            </style>
            """,
            unsafe_allow_html=True,
        )

# Load CSS
load_css()

# ------------- Enhanced Sidebar Navigation ------------- 
with st.sidebar:
    # Logo and title section
    col1, col2 = st.columns([1, 3])
    with col1:
        st.image(
            "https://cdn-icons-png.flaticon.com/512/3209/3209265.png", 
            width=50
        )
    with col2:
        st.markdown("### 🧬 **Cell Analysis Hub**")
    
    st.markdown("---")
    
    # Dark mode toggle with enhanced styling
    dark_label = "🌙 Dark Mode" if not st.session_state.dark_mode else "☀️ Light Mode"
    if st.button(dark_label, key="mode_toggle"):
        toggle_mode()
        st.rerun()
    
    st.markdown("---")
    
    # Navigation with categories
    st.markdown("### 📊 **Navigation**")
    
    # Core Analysis Features
    st.markdown("**🔬 Core Analysis**")
    analysis_options = [
        "🏠 Home",
        "🔍 Single Image Prediction", 
        "📊 Batch Analysis",
        "📈 Metrics & Evaluation"
    ]
    
    # Tools & Utilities
    st.markdown("**🛠️ Tools & Utilities**")
    tools_options = [
        "🖼️ Example Images",
        "⚙️ OpenCV Tools", 
        "🔢 Cell Counting",
        "⏱️ Time-lapse Analysis"
    ]
    
    # Information & Support
    st.markdown("**ℹ️ Information & Support**")
    info_options = [
        "📝 Prediction History",
        "🤖 Model Info",
        "💬 Researcher Chatbot"
    ]
    
    # Combined page selection
    all_pages = [
        "Home",
        "Single Image Prediction",
        "Batch Analysis", 
        "Metrics & Evaluation",
        "Example Images",
        "OpenCV Tools",
        "Cell Counting",
        "Time-lapse Analysis", 
        "Prediction History",
        "Model Info",
        "Researcher Chatbot"
    ]
    
    page = st.selectbox(
        "Choose a feature:",
        all_pages,
        help="Select the analysis tool or feature you want to use",
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    
    # Enhanced Feature Guide
    with st.expander("📚 **Feature Guide & Documentation**", expanded=False):
        st.markdown("""
        <div class="feature-card">
        
        ### 🏠 **Core Analysis Features**
        
        **🏠 Home:** Overview and comprehensive usage instructions for the application.
        
        **🔍 Single Image Prediction:** Upload individual cell images for instant AI-powered classification (benign/malignant) with confidence scores.
        
        **📊 Batch Analysis:** Process multiple images simultaneously via CSV upload, with statistical distributions and exportable results.
        
        **📈 Metrics & Evaluation:** Comprehensive model performance evaluation with ground-truth comparison, confusion matrices, and detailed metrics.
        
        ### 🛠️ **Advanced Tools**
        
        **🖼️ Example Images:** Browse curated sample images from both benign and malignant cell datasets for reference.
        
        **⚙️ OpenCV Tools:** Interactive image processing suite with real-time filters, transformations, and augmentation capabilities.
        
        **🔢 Cell Counting:** Automated cell detection and counting with size distribution analysis for microscopy images.
        
        **⏱️ Time-lapse Analysis:** Track cell population changes over time sequences with trend analysis.
        
        ### 📋 **Data Management**
        
        **📝 Prediction History:** Review, analyze, and export your complete session prediction history.
        
        **🤖 Model Info:** Detailed model architecture, training data specifications, and performance benchmarks.
        
        **💬 Researcher Chatbot:** AI-powered assistant for medical research questions and methodology guidance.
        
        </div>
        """, unsafe_allow_html=True)
    
    # Additional sidebar footer
    st.markdown("---")
    st.markdown("""
        <div style="text-align: center; padding: 1rem; color: #64748b; font-size: 0.8rem;">
            <strong>Cell Analysis Dashboard v2.0</strong><br>
            Powered by Advanced AI Models<br>
            🔬 Professional Medical Research Tool
        </div>
    """, unsafe_allow_html=True)

# Main content area enhancement
st.markdown("""
    <div class="dashboard-title">
        🧬 Cell Analysis Dashboard
    </div>
    <div class="subtitle">
        Advanced AI-Powered Cell Classification & Analysis Platform
    </div>
""", unsafe_allow_html=True)

# Display current page info
if page != "Home":
    st.markdown(f"""
        <div class="feature-card">
            <h3>📍 Current Page: {page}</h3>
            <p>You are now in the <strong>{page}</strong> section. Use the sidebar navigation to switch between different features and tools.</p>
        </div>
    """, unsafe_allow_html=True)

# ------------------ PUBMED (NCBI) LITERATURE SEARCH ------------------
def search_pubmed(query, max_results=5):
    """
    Search PubMed for articles related to the given query using Entrez ESearch & ESummary.
    Returns a list of dictionaries: [{title, url, summary, pubdate, authors}]
    """
    # Step 1: ESearch
    base_search = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    params = {
        "db": "pubmed",
        "term": query,
        "retmax": max_results,
        "retmode": "json"
    }
    r = requests.get(base_search, params=params)
    if r.status_code != 200:
        return []
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    if not ids:
        return []

    # Step 2: ESummary
    base_summary = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    params = {
        "db": "pubmed",
        "id": ",".join(ids),
        "retmode": "xml"
    }
    r = requests.get(base_summary, params=params)
    if r.status_code != 200:
        return []

    # Parse XML
    articles = []
    root = ET.fromstring(r.content)
    for doc in root.findall(".//DocSum"):
        title = doc.findtext("Item[@Name='Title']", default="No Title")
        pubdate = doc.findtext("Item[@Name='PubDate']", default="No Date")
        source = doc.findtext("Item[@Name='Source']", default="PubMed")
        authors = [item.text for item in doc.findall("Item[@Name='AuthorList']/Item")]
        article_id = doc.findtext("Id", default="")
        url = f"https://pubmed.ncbi.nlm.nih.gov/{article_id}/"
        articles.append({
            "title": title,
            "url": url,
            "summary": source,
            "pubdate": pubdate,
            "authors": ", ".join(authors)
        })
    return articles

def pubmed_search_box(prediction, extra_query=None):
    st.markdown("### 🔎 Related Biomedical Literature Search (PubMed)")
    st.write("Find recent research articles relevant to the model's prediction or your custom query.")
    default_query = prediction if extra_query is None else f"{prediction} {extra_query}"
    user_query = st.text_input("Enter search keywords for PubMed:", value=default_query)
    if st.button("Search PubMed"):
        with st.spinner("Searching PubMed..."):
            articles = search_pubmed(user_query)
            if not articles:
                st.warning("No results found.")
            else:
                for a in articles:
                    st.markdown(f"**[{a['title']}]({a['url']})**  \n"
                                f"*{a['pubdate']}*  \n"
                                f"Authors: {a['authors'] or 'N/A'}")
                    st.markdown("---")

# ------------- Core Functions -------------
@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(pretrained=False)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load("best_model.pth", map_location=device))
    model.eval()
    model.to(device)
    return model, device

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

def predict_image(img, model, device, threshold=0.5):
    transform = get_transform()
    img_tensor = transform(img).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(img_tensor)
        prob = torch.sigmoid(output).item()
        pred = "Malignant" if prob > threshold else "Benign"
    return pred, prob

def show_example_images():
    st.subheader("📷 Example Images from Dataset")
    base_dirs = [("Benign", "data/benign"), ("Malignant", "data/malignant")]
    cols = st.columns(2)
    for i, (label, path) in enumerate(base_dirs):
        if os.path.exists(path):
            images = [f for f in os.listdir(path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))]
            if images:
                img_file = random.choice(images)
                img = Image.open(os.path.join(path, img_file))
                cols[i].image(img, caption=f"{label} example", use_container_width=True)
            else:
                cols[i].warning(f"No images found in {path}")

def save_history_to_csv(history):
    df = pd.DataFrame(history)
    csv = df.to_csv(index=False).encode('utf-8')
    return csv

def get_model_info():
    info = {
        "Model": "ResNet-18",
        "Trained on": "Custom dataset (benign/malignant)",
        "Last updated": datetime.fromtimestamp(os.path.getmtime("best_model.pth")).strftime('%Y-%m-%d %H:%M'),
        "Threshold": 0.5,
    }
    return info

def analyze_batch(images, model, device, threshold=0.5):
    results = []
    for img in images:
        pred, prob = predict_image(img, model, device, threshold)
        results.append({"Result": pred, "Confidence": prob})
    return results

def plot_distribution(df):
    plt.figure(figsize=(4,2))
    ax = sns.countplot(x="Result", data=df)
    plt.title("Prediction Distribution")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption="Prediction Distribution")
    plt.close()

def plot_confidence_hist(df):
    plt.figure(figsize=(4,2))
    sns.histplot(df["Confidence"], bins=10, kde=True)
    plt.title("Confidence Score Distribution")
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png")
    buf.seek(0)
    st.image(buf, caption="Confidence Score Distribution")
    plt.close()

# ------------- OpenCV Tools Feature -------------
def opencv_tools():
    st.header("🧪 OpenCV Image Processing Tools")
    st.write("Upload an image and preview various OpenCV-based transformations and preprocessings.")
    uploaded = st.file_uploader("Upload an image for OpenCV tools", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)
        st.image(image, caption="Original Image", use_container_width=True)

        st.subheader("Select OpenCV Operation")
        operation = st.selectbox("Operation", [
            "Grayscale",
            "Gaussian Blur",
            "Canny Edge Detection",
            "Adaptive Threshold",
            "Contour Detection",
            "CLAHE (Contrast Limited Adaptive Histogram Equalization)",
            "Image Rotation",
            "Horizontal Flip",
            "Vertical Flip",
            "Add Gaussian Noise"
        ])

        if operation == "Grayscale":
            processed = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            st.image(processed, caption="Grayscale", channels="GRAY", use_container_width=True)

        elif operation == "Gaussian Blur":
            ksize = st.slider("Kernel Size", 1, 21, 5, step=2)
            processed = cv2.GaussianBlur(img_np, (ksize, ksize), 0)
            st.image(processed, caption=f"Gaussian Blur (ksize={ksize})", use_container_width=True)

        elif operation == "Canny Edge Detection":
            th1 = st.slider("Threshold1", 0, 255, 100)
            th2 = st.slider("Threshold2", 0, 255, 200)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, th1, th2)
            st.image(edges, caption=f"Canny Edges (T1={th1}, T2={th2})", channels="GRAY", use_container_width=True)

        elif operation == "Adaptive Threshold":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            block = st.slider("Block Size", 3, 99, 11, step=2)
            C = st.slider("C (subtract)", 0, 20, 2)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                           cv2.THRESH_BINARY, block, C)
            st.image(thresh, caption=f"Adaptive Threshold (block={block}, C={C})", channels="GRAY", use_container_width=True)

        elif operation == "Contour Detection":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            ret, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            contour_img = img_np.copy()
            cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
            st.image(contour_img, caption="Contour Detection", use_container_width=True)

        elif operation == "CLAHE (Contrast Limited Adaptive Histogram Equalization)":
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            clip = st.slider("Clip Limit", 1, 10, 2)
            tile = st.slider("Tile Grid Size", 1, 16, 8)
            clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
            cl1 = clahe.apply(gray)
            st.image(cl1, caption="CLAHE", channels="GRAY", use_container_width=True)

        elif operation == "Image Rotation":
            angle = st.slider("Rotate Angle", -180, 180, 0)
            height, width = img_np.shape[:2]
            M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
            rotated = cv2.warpAffine(img_np, M, (width, height))
            st.image(rotated, caption=f"Rotated ({angle} deg)", use_container_width=True)

        elif operation == "Horizontal Flip":
            flipped = cv2.flip(img_np, 1)
            st.image(flipped, caption="Horizontally Flipped", use_container_width=True)

        elif operation == "Vertical Flip":
            flipped = cv2.flip(img_np, 0)
            st.image(flipped, caption="Vertically Flipped", use_container_width=True)

        elif operation == "Add Gaussian Noise":
            mean = st.slider("Mean", 0, 100, 0)
            std = st.slider("Std Dev", 1, 100, 10)
            noise = np.random.normal(mean, std, img_np.shape).astype(np.int16)
            noisy = np.clip(img_np + noise, 0, 255).astype(np.uint8)
            st.image(noisy, caption=f"Gaussian Noise (mean={mean}, std={std})", use_container_width=True)

# ----------- Automated Cell Counting & Size Distribution -----------
def automated_cell_counting():
    st.header("🔬 Automated Cell Counting & Size Distribution")
    st.write("Upload a microscopy image. The tool will segment, count cells, and plot the cell size distribution.")

    uploaded = st.file_uploader("Upload a microscopy/cell image", type=["png", "jpg", "jpeg"])
    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        img_np = np.array(image)
        st.image(image, caption="Original Image", use_container_width=True)

        # --- Preprocessing & Cell Segmentation ---
        gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
        blur = cv2.medianBlur(gray, 5)
        ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        kernel = np.ones((3,3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)

        # --- Find contours (cells) ---
        contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cell_areas = [cv2.contourArea(cnt) for cnt in contours if cv2.contourArea(cnt) > 10]

        count = len(cell_areas)
        st.success(f"Cells Detected: {count}")

        # --- Draw contours on original image ---
        contour_img = img_np.copy()
        cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
        st.image(contour_img, caption="Cell Contours", use_container_width=True)

        # --- Plot area distribution ---
        if cell_areas:
            fig, ax = plt.subplots()
            ax.hist(cell_areas, bins=15, color='skyblue', edgecolor='black')
            ax.set_xlabel("Cell Area (pixels)")
            ax.set_ylabel("Count")
            ax.set_title("Cell Size Distribution")
            st.pyplot(fig)
        else:
            st.info("No cells detected above minimum area threshold.")

# ----------- Time-lapse / Sequence Analysis -----------
def time_lapse_sequence_analysis():
    st.header("⏳ Time-lapse / Sequence Analysis")
    st.write("Upload a sequence of images (same size, ordered by time). Analyze cell changes over time.")

    uploaded_files = st.file_uploader("Upload a sequence of images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)
    if uploaded_files:
        uploaded_files = sorted(uploaded_files, key=lambda x: x.name)  # Try to order by filename
        cell_counts = []
        fig, ax = plt.subplots(1, len(uploaded_files), figsize=(4*len(uploaded_files), 4))
        if len(uploaded_files) == 1:
            ax = [ax]

        for i, file in enumerate(uploaded_files):
            image = Image.open(file).convert("RGB")
            img_np = np.array(image)
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            blur = cv2.medianBlur(gray, 5)
            ret, thresh = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
            contours, _ = cv2.findContours(opening, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            count = len([cnt for cnt in contours if cv2.contourArea(cnt) > 10])
            cell_counts.append(count)
            contour_img = img_np.copy()
            cv2.drawContours(contour_img, contours, -1, (0,255,0), 2)
            ax[i].imshow(contour_img)
            ax[i].set_title(f"Frame {i+1} (Cells: {count})")
            ax[i].axis('off')
        st.pyplot(fig)

        # --- Plot cell count vs time ---
        fig2, ax2 = plt.subplots()
        ax2.plot(range(1, len(cell_counts)+1), cell_counts, marker='o')
        ax2.set_xlabel("Frame (Time)")
        ax2.set_ylabel("Cell Count")
        ax2.set_title("Cell Count Over Time")
        st.pyplot(fig2)

# ------------- Session State -------------
if "history" not in st.session_state:
    st.session_state["history"] = []

# ------------- Page Routing -------------
if page == "Home":
   # st.title("🧬Deep Learning-Based Detection of Malignant and Benign Cells from Microscopic Images")
    st.markdown("""
<div style="font-size:18px">
Welcome to the <b>Cancer Cell Classification Platform</b>!  
This AI-powered suite leverages deep learning to help users and researchers classify cell images as <b>benign</b> or <b>malignant</b>, and provides a professional assistant for general medical research questions.

<hr style="margin-top:0.2em;margin-bottom:0.2em;"/>

<b>Features Overview:</b>
- <b>Single Image Prediction</b>: Upload a cell image for instant prediction.
- <b>Batch Analysis</b>: Run predictions on large sets of images via CSV, with distributions and export.
- <b>Metrics & Evaluation</b>: Upload ground-truth labels for metrics & confusion matrix.
- <b>Example Images</b>: Browse random benign/malignant samples.
- <b>OpenCV Tools</b>: Visualize and interactively apply image processing and augmentation using OpenCV.
- <b>Cell Counting</b>: Automated cell counting and size distribution analysis for microscopy images.
- <b>Time-lapse Analysis</b>: Analyze cell count changes in a sequence of images.
- <b>Session Prediction History</b>: Export/download all predictions.
- <b>Model Info</b>: View model architecture, dataset, and last update.
- <b>Researcher Chatbot</b>: Ask general medical research questions and get expert answers.

<b>How to Use:</b>
<ol>
<li>Select a feature from the sidebar dashboard.</li>
<li>For predictions: Upload an image or CSV as instructed.</li>
<li>View results: The app displays predictions, confidence scores, and visual summaries.</li>
<li>For metrics: Upload ground-truth CSV for evaluation.</li>
<li>Export: Download your prediction history or batch results as CSV.</li>
<li>Chatbot: Go to the Researcher Chatbot tab for medical research Q&A.</li>
</ol>

<div style="color:#d9534f;"><b>Note:</b> This platform is for research and educational purposes only, and should not be used for clinical decision making.</div>
</div>
""", unsafe_allow_html=True)

elif page == "Single Image Prediction":
    st.header("🔎 Single Image Prediction")
    st.write("Upload a single cell image to obtain a prediction (benign or malignant).")
    uploaded_file = st.file_uploader("Choose a cell image...", type=["png", "jpg", "jpeg"])
    model, device = load_model()
    if uploaded_file is not None:
        with st.spinner("Analyzing image..."):
            img = Image.open(uploaded_file).convert("RGB")
            pred, prob = predict_image(img, model, device)
            st.image(img, caption="Uploaded Image", use_container_width=True)
            st.success(f"**Prediction:** {pred}")
            st.info(f"Confidence: `{prob:.4f}`")
            st.session_state["history"].append({
                "Time": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                "Result": pred,
                "Confidence": f"{prob:.4f}"
            })
            # --- Integrated Literature Search ---
            pubmed_search_box(prediction=pred, extra_query="cancer cell")

elif page == "Batch Analysis":
    st.header("🧪 Batch Prediction & Distribution Analysis")
    st.write(
        "Upload a CSV with a column `image_path` listing paths to images for batch prediction. "
        "You will see result tables, prediction distributions, and can export results."
    )
    csv_file = st.file_uploader("Upload CSV for Batch Analysis", type=["csv"])
    batch_df = None
    model, device = load_model()
    if csv_file is not None:
        try:
            df = pd.read_csv(csv_file)
            img_col = "image_path" if "image_path" in df.columns else df.columns[0]
            df = df.dropna(subset=[img_col])
            df[img_col] = df[img_col].astype(str)
            images, valid_paths, invalid_paths = [], [], []
            for p in df[img_col]:
                if os.path.exists(p):
                    images.append(Image.open(p).convert("RGB"))
                    valid_paths.append(p)
                else:
                    invalid_paths.append(p)
            if images:
                results = analyze_batch(images, model, device)
                batch_df = pd.DataFrame({
                    "image_path": valid_paths,
                    "Result": [r["Result"] for r in results],
                    "Confidence": [r["Confidence"] for r in results]
                })
                st.session_state["batch_df"] = batch_df
                st.write("### Batch Results")
                st.dataframe(batch_df, use_container_width=True)
                st.markdown("#### Distribution Visualizations")
                plot_distribution(batch_df)
                plot_confidence_hist(batch_df)
                csv = batch_df.to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="⬇️ Download Batch Results as CSV",
                    data=csv,
                    file_name="batch_prediction_results.csv",
                    mime="text/csv"
                )
                if not batch_df.empty and st.checkbox("Search PubMed for results in this batch"):
                    unique_preds = batch_df["Result"].unique().tolist()
                    selected = st.selectbox("Select prediction to search PubMed:", unique_preds)
                    pubmed_search_box(prediction=selected, extra_query="cancer cell")
            if invalid_paths:
                st.warning(f"Some images could not be found/read: {invalid_paths}")
        except Exception as e:
            st.error(f"Batch analysis failed: {e}")

elif page == "Metrics & Evaluation":
    st.header("📊 Metrics & Evaluation")
    st.write(
        "Upload a CSV with columns `image_path` and `true_label` (Benign/Malignant) after a batch prediction run to view metrics."
    )
    metrics_csv = st.file_uploader("Upload CSV for Metrics Evaluation", type=["csv"])
    if metrics_csv is not None:
        try:
            metrics_df = pd.read_csv(metrics_csv)
            if "batch_df" not in st.session_state:
                st.error("Please run batch analysis first and keep this tab open.")
            else:
                batch_df = st.session_state["batch_df"]
                merged = batch_df.merge(metrics_df, on="image_path", how="inner")
                y_true = merged["true_label"]
                y_pred = merged["Result"]
                from sklearn.metrics import classification_report, confusion_matrix
                report = classification_report(y_true, y_pred, output_dict=True)
                st.write("### Classification Report")
                st.json(report)
                cm = confusion_matrix(y_true, y_pred, labels=["Benign", "Malignant"])
                fig, ax = plt.subplots()
                sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Benign", "Malignant"], yticklabels=["Benign", "Malignant"], ax=ax)
                ax.set_xlabel("Predicted")
                ax.set_ylabel("True")
                st.pyplot(fig)
        except Exception as e:
            st.error(f"Metrics evaluation failed: {e}")

elif page == "Example Images":
    st.header("🖼️ Example Images")
    show_example_images()

elif page == "OpenCV Tools":
    opencv_tools()

elif page == "Cell Counting":
    automated_cell_counting()

elif page == "Time-lapse Analysis":
    time_lapse_sequence_analysis()

elif page == "Prediction History":
    st.header("🗒️ Prediction History")
    user_history = st.session_state["history"]
    if user_history:
        df = pd.DataFrame(user_history)
        st.dataframe(df, use_container_width=True)
        csv = save_history_to_csv(user_history)
        st.download_button(
            label="⬇️ Download History as CSV",
            data=csv,
            file_name="prediction_history.csv",
            mime="text/csv"
        )
        if st.button("🧹 Clear History"):
            st.session_state["history"] = []
            st.experimental_rerun()
    else:
        st.info("No predictions yet. Make a prediction to see history here.")

elif page == "Model Info":
    st.header("ℹ️ Model Info")
    info = get_model_info()
    for k, v in info.items():
        st.write(f"**{k}:** {v}")

elif page == "Researcher Chatbot":
    st.title("🤖 Medical Researcher Chatbot Assistant")
    st.markdown("""
<div style="font-size:17px">
This chatbot assists medical researchers with <b>general medical questions</b> relevant to research and clinical practice.

<b>Ask any question related to:</b>
<ul>
<li>General medical knowledge and terminology</li>
<li>Disease mechanisms, diagnosis, and treatment</li>
<li>Medical imaging and diagnostic techniques</li>
<li>Laboratory methods, data analysis, and interpretation</li>
<li>Clinical guidelines, research best practices, and scientific advances</li>
<li>Medical statistics, study design, and publication standards</li>
</ul>
</div>
---
""", unsafe_allow_html=True)
    # Gemini API setup
    GEMINI_API_KEY = ""
    GEMINI_ENDPOINT = f""

    def get_gemini_reply(user_input):
        research_prompt = """
You are an expert AI assistant specializing in medical research. 
You help medical researchers by answering questions on a wide range of medical topics, including but not limited to:

- General medical knowledge and terminology
- Disease mechanisms, diagnosis, and treatment
- Medical imaging and diagnostic techniques
- Laboratory methods, data analysis, and interpretation
- Clinical guidelines, research best practices, and recent scientific advances
- Medical statistics, study design, and publication standards

Provide responses that are accurate, clear, and tailored for a medical research audience. 
If a question requires clinical judgment, advise consulting qualified healthcare professionals, and do not provide medical diagnoses or treatment for individual patients.
"""
        payload = {
            "contents": [{
                "parts": [{
                    "text": f"{research_prompt}\n\nUser: {user_input}\n\nAI:"
                }]
            }]
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(GEMINI_ENDPOINT, json=payload, headers=headers)

        if response.status_code == 200:
            try:
                reply = response.json()["candidates"][0]["content"]["parts"][0]["text"]
                # Clean up formatting for Streamlit
                reply = reply.replace("• ", "\n- ")
                reply = reply.replace("* ", "\n- ")
                return reply.strip()
            except Exception:
                return "Sorry, I couldn't parse the response from the AI model."
        else:
            return "API request failed. Please try again later."

    # Chat interface
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    with st.form("chat_form", clear_on_submit=True):
        user_input = st.text_area(
            "Ask your question about any area of medical research:",
            height=70,
            placeholder="E.g., What are the latest advances in cancer immunotherapy?"
        )
        submitted = st.form_submit_button("Ask")

    if submitted and user_input.strip():
        with st.spinner("Thinking..."):
            reply = get_gemini_reply(user_input.strip())
            st.session_state.chat_history.append(("You", user_input.strip()))
            st.session_state.chat_history.append(("Assistant", reply))

    # Display chat history
    for speaker, message in st.session_state.chat_history:
        if speaker == "You":
            st.markdown(f"<div style='color:#1f77b4;'><b>You:</b> {message}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"<div style='color:#2ca02c;'><b>Assistant:</b> {message}</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.caption("Powered by Gemini | For research and educational purposes only.")

st.markdown("<hr/>", unsafe_allow_html=True)
st.caption("Developed by leksialevin7700 | Cancer Cell Classifier Demo | All rights reserved © 2025")