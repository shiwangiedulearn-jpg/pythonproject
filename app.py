import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from datetime import datetime
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import base64

st.set_page_config(page_title="AI Career Advisor", layout="wide")

# ---------- SET BACKGROUND IMAGE FROM LOCAL ---------- #
def set_bg_from_local(image_path):
    with open(image_path, "rb") as img_file:
        b64_string = base64.b64encode(img_file.read()).decode()
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/png;base64,{b64_string}");
        background-size: cover;
        background-attachment: fixed;
        background-position: center;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

set_bg_from_local("background.jpg")  

# ---------- CUSTOM STYLE ---------- #
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@500&display=swap');

    html, body, [class*="css"] {
        font-family: 'Orbitron', sans-serif;
        color: #ffffff;
    }
   /* Sidebar container */
    [data-testid="stSidebar"] {
        background: rgba(0, 0, 0, 0.6);  /* translucent dark */
        color: #ffffff;
        min-width:400px;
        padding: 2rem;
        border-right: 1px solid rgba(0, 255, 224, 0.3);
    }

    /* Sidebar title */
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: #00ffe0;
        font-weight: bold;
    }

    /* Sidebar paragraph text */
    [data-testid="stSidebar"] p {
        font-size: 0.95rem;
        color: #e6e6e6;
    }

    /* Quote box inside sidebar */
    .stAlert {
        background-color: rgba(0, 255, 224, 0.1) !important;
        color: #00ffe0 !important;
        border-radius: 10px;
        padding: 10px;
        font-size: 0.9rem;
    }
    .glass-panel {
        background: rgba(255, 255, 255, 0.1);
        border-radius: 15px;
        padding: 2rem;
        box-shadow: 0 8px 32px 0 rgba(31, 38, 135, 0.37);
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
    }

    /* Normal buttons */
    /* Default button style (for right side main content) */
.stButton>button {
    background-color: #000000 !important;  /* Black background */
    color: #ffffff !important;             /* White text */
    font-weight: bold !important;
    border-radius: 12px !important;
    padding: 0.6rem 1.2rem !important;
    border: none !important;
    transition: 0.3s !important;
}

/* Hover effect */
.stButton>button:hover {
    background-color: #222222 !important;  /* Slightly lighter black */
    color: #ffffff !important;
}

/* Sidebar Get Advice button (cyan) */
section[data-testid="stSidebar"] div.stButton > button {
    background-color: #00ccaa !important;
    color: #000000 !important;
    font-weight: bold !important;
    border-radius: 12px !important;
    border: none !important;
}
section[data-testid="stSidebar"] div.stButton > button:hover {
    background-color: #00bfa5 !important;
    color: #000000 !important;
}


    /* Force sidebar Get Advice button text to black */
    section[data-testid="stSidebar"] div.stButton > button {
        color: #000000 !important;
        font-weight: bold !important;
        background-color: #00ccaa !important;
        border-radius: 12px !important;
    }
    section[data-testid="stSidebar"] div.stButton > button:hover {
        background-color: #00ccaa !important;
        color: black !important;
    }

    .stRadio>div>label {
        background-color: rgba(0,0,0,0.6);
        padding: 0.4rem 1rem;
        border-radius: 12px;
    }

    .title h1 {
        color: #00ffe0;
        font-size: 3rem;
        font-weight: bold;
    }

    thead tr th {
        background-color: rgba(0, 255, 224, 0.15) !important;
        color: #00ffe0 !important;
        font-weight: bold;
        border-bottom: 1px solid #00ffe0;
    }
    tbody tr td {
        background-color: rgba(255, 255, 255, 0.05) !important;
        color: #ffffff !important;
        border-bottom: 0.5px solid rgba(255, 255, 255, 0.1);
    }
    tbody tr:hover {
        background-color: rgba(0, 255, 224, 0.1) !important;
    }
    .css-1l269bu {
        border: none !important;
        border-radius: 12px !important;
        overflow: hidden !important;
    }
    /* Make all main content text white (except sidebar) */
.main-content, .stApp > div:nth-child(1) {
    color: white !important;
}

/* Ensure specific text areas like tables, headings, and markdown stay white */
[data-testid="stMarkdownContainer"], 
[data-testid="stHeader"], 
[data-testid="stSubheader"],
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    color: white !important;
}

/* Keep sidebar text styling intact */
[data-testid="stSidebar"] {
    color: #e6e6e6 !important;
}

    </style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    df = pd.read_csv("college_student_placement_dataset.csv")
    df.drop_duplicates(subset="College_ID", inplace=True)
    df["Internship_Experience"] = df["Internship_Experience"].map({"Yes": 1, "No": 0})
    df["Placement"] = df["Placement"].map({"Yes": 1, "No": 0})
    return df

df = load_data()

X = df.drop(columns=["College_ID", "Placement"])
y = df["Placement"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
model_accuracy = accuracy_score(y_test, y_pred)

# ---------- SIDEBAR AI MENTOR (Fixed AI Response) ---------- #
st.sidebar.title("🤖 AI Mentor")
st.sidebar.markdown("Hi there! I'm your career advisor bot. Input your scores and I’ll predict placement & give tips!")

# New text for placement problem solving
st.sidebar.info("💬 Enter your problems related to your placement and I will help you out")

# Input for student's placement question
user_question = st.sidebar.text_area(
    "Ask me anything about your placement:", 
    placeholder="e.g., How can I improve my CGPA impact on placements?"
)

# Button to trigger AI response
if st.sidebar.button("Get Advice"):
    # only run if user provided a question
    if user_question and user_question.strip():
        try:
            import os

            # Read Groq API key from environment to avoid committing secrets to source control
            groq_api_key = os.getenv("GROQ_API_KEY")
            if not groq_api_key:
                raise Exception(
                    "GROQ_API_KEY environment variable is not set. Set it before running the app."
                )

            llm = ChatOpenAI(
                model_name="llama-3.1-8b-instant",
                temperature=0.5,
                openai_api_key=groq_api_key,
                openai_api_base="https://api.groq.com/openai/v1",
            )

            response = llm([HumanMessage(content=user_question)])
            answer = response.content if hasattr(response, "content") else str(response)

            st.sidebar.success(f"AI Mentor: {answer}")

        except Exception as e:
            st.sidebar.error(f"⚠️ Unable to fetch AI response: {str(e)}")
    else:
        st.sidebar.warning("Please enter a question to get advice.")


st.markdown('<div class="glass-panel">', unsafe_allow_html=True)
st.markdown("<div class='title'><h1>🧠 AI Career Advisor</h1></div>", unsafe_allow_html=True)
st.markdown("""
Welcome to your futuristic placement predictor and career assistant. Enter your stats and receive:
- A prediction about your placement chances 🎯
- Key performance metrics 📊
- Smart suggestions on how to improve 🧠
""")
st.divider()

page = st.radio("Go to", ["Raw Data", "Visualizations", "Model & Prediction"], horizontal=True)


if page == "Raw Data":
    st.subheader("📄 Student Dataset Overview")

    # 1️⃣ Main Dataset Styling
    numeric_df = df.select_dtypes(include='number')
    vmin = numeric_df.min().min()
    vmax = numeric_df.max().max()
    cmap = plt.get_cmap('Greys')  # Darker grey colormap

    def style_cell(val):
        try:
            norm_val = (float(val) - vmin) / (vmax - vmin)  # normalize value
            color = mcolors.to_hex(cmap(norm_val))
            # Compute brightness to decide text color
            rgb = mcolors.hex2color(color)
            brightness = 0.299*rgb[0] + 0.587*rgb[1] + 0.114*rgb[2]
            text_color = 'white' if brightness < 0.5 else 'black'
        except:
            color = '#39383C'
            text_color = 'white'
        return f'background-color: {color}; color: {text_color}; '

    # Apply styling
    styled_df = df.style.applymap(style_cell)
    styled_df = styled_df.applymap(
        lambda v: 'background-color:#39383C;color:white;font-weight:bold;', 
        subset=['College_ID']
    )

    st.write(styled_df)

    # 2️⃣ Basic Stats Table
    st.markdown("### Basic Stats")
    styled_stats = df.describe().style.background_gradient(cmap='Greys')
    st.write(styled_stats)




elif page == "Visualizations":
    st.subheader("📊 Data Visualizations")
    placement_counts = df["Placement"].value_counts().rename(index={0: "Not Placed", 1: "Placed"})

    fig_bar = px.bar(
        x=["Not Placed", "Placed"],
        y=placement_counts.values,
        color=["Not Placed", "Placed"],
        color_discrete_map={"Not Placed": "#A80CF1", "Placed": "#A80CF1"},
        labels={"x": "Placement Status", "y": "Count"},
        title="Placement Distribution"
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    fig1 = px.histogram(df, x="CGPA", color="Placement", barmode="overlay",
                        color_discrete_map={0: "#2B07AE", 1: "#C99AF3"})
    st.plotly_chart(fig1, use_container_width=True)

    perf_placement = df.groupby("Academic_Performance")["Placement"].mean()
    fig_line = px.line(
        x=perf_placement.index,
        y=perf_placement.values,
        labels={"x": "Academic Performance", "y": "Placement Rate"},
        title="Academic Performance vs Placement",
        markers=True,
        line_shape="linear"
    )
    fig_line.update_traces(line=dict(color="#A80CF1", width=3))
    st.plotly_chart(fig_line, use_container_width=True)

    st.subheader("Feature Correlation Matrix")
    st.dataframe(df.select_dtypes(include=np.number).corr().style.background_gradient(cmap='magma'))

elif page == "Model & Prediction":
    st.subheader("🔍 Predict Your Placement Chance")
    st.metric("📈 Model Accuracy", f"{model_accuracy*100:.2f}%")

    with st.form("placement_form"):
        col1, col2, col3 = st.columns(3)
        with col1:
            iq = st.slider("IQ", 70, 140, 100)
            prev_sem = st.slider("Previous Semester Result", 4.0, 10.0, 6.5)
            cgpa = st.slider("Current CGPA", 4.0, 10.0, 7.5)
        with col2:
            academic = st.slider("Academic Performance", 1, 10, 6)
            internship = st.selectbox("Internship Experience", ["Yes", "No"])
            ec_score = st.slider("Extra Curricular Score", 0, 10, 5)
        with col3:
            comm = st.slider("Communication Skills", 0, 10, 6)
            projects = st.slider("Projects Completed", 0, 5, 2)

        submitted = st.form_submit_button("🚀 Predict My Placement")

    if submitted:
        input_data = pd.DataFrame({
            "IQ": [iq],
            "Prev_Sem_Result": [prev_sem],
            "CGPA": [cgpa],
            "Academic_Performance": [academic],
            "Internship_Experience": [1 if internship == "Yes" else 0],
            "Extra_Curricular_Score": [ec_score],
            "Communication_Skills": [comm],
            "Projects_Completed": [projects]
        })

        result = model.predict(input_data)[0]
        prediction = "✅ Likely to Get Placed" if result == 1 else "❌ May Need to Improve"
        emoji = "🎉" if result == 1 else "🔧"

        st.success(f"{emoji} {prediction}")

        if result == 0:
            st.info("📌 Tips: Work on internships, build portfolio, and practice interviews!")
        else:
            st.balloons()

        st.markdown(f"_Prediction generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}_")
