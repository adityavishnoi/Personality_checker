import streamlit as st
import pandas as pd
import numpy as np
import pickle

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Personality AI",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CUSTOM CSS
# ============================================================
st.markdown("""
<style>

@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

* {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background:
        radial-gradient(circle at 10% 10%, rgba(124, 58, 237, 0.18), transparent 25%),
        radial-gradient(circle at 90% 10%, rgba(37, 99, 235, 0.15), transparent 25%),
        radial-gradient(circle at 50% 100%, rgba(236, 72, 153, 0.12), transparent 30%),
        #070b16;
    color: white;
}

.block-container {
    max-width: 1250px;
    padding-top: 2rem;
    padding-bottom: 4rem;
}

/* ================= HERO ================= */

.hero {
    padding: 45px 25px;
    border-radius: 28px;
    text-align: center;

    background:
        linear-gradient(
            135deg,
            rgba(124, 58, 237, 0.23),
            rgba(37, 99, 235, 0.18),
            rgba(236, 72, 153, 0.18)
        );

    border: 1px solid rgba(255,255,255,0.10);

    box-shadow:
        0 20px 60px rgba(0,0,0,0.35);

    margin-bottom: 30px;
}

.hero-title {
    font-size: 3.5rem;
    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #a78bfa,
            #60a5fa,
            #f472b6
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.hero-subtitle {
    color: #cbd5e1;
    font-size: 1.05rem;
    margin-top: 8px;
}

/* ================= SECTION CARDS ================= */

.section-header {
    margin-top: 28px;
    margin-bottom: 15px;
}

.section-title {
    font-size: 1.45rem;
    font-weight: 700;
}

.section-description {
    color: #94a3b8;
    font-size: 0.9rem;
    margin-top: 4px;
}

/* ================= SLIDERS ================= */

.stSlider {
    padding-bottom: 8px;
}

/* ================= BUTTON ================= */

.stButton > button {
    width: 100%;

    padding: 15px;

    border-radius: 15px;

    border: none;

    background:
        linear-gradient(
            90deg,
            #7c3aed,
            #2563eb,
            #ec4899
        );

    color: white;

    font-size: 1rem;
    font-weight: 700;

    box-shadow:
        0 10px 30px rgba(124,58,237,0.3);

    transition: 0.25s ease;
}

.stButton > button:hover {
    transform: translateY(-2px);

    box-shadow:
        0 15px 40px rgba(124,58,237,0.45);
}

/* ================= RESULT ================= */

.result-box {
    margin-top: 30px;

    padding: 40px 20px;

    text-align: center;

    border-radius: 25px;

    background:
        linear-gradient(
            135deg,
            rgba(124,58,237,0.27),
            rgba(37,99,235,0.20),
            rgba(236,72,153,0.20)
        );

    border: 1px solid rgba(255,255,255,0.12);

    box-shadow:
        0 20px 60px rgba(124,58,237,0.18);
}

.result-small {
    color: #cbd5e1;
    font-size: 1rem;
}

.result-big {
    margin-top: 8px;

    font-size: 3rem;

    font-weight: 800;

    background:
        linear-gradient(
            90deg,
            #a78bfa,
            #60a5fa,
            #f472b6
        );

    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

/* ================= SIDEBAR ================= */

section[data-testid="stSidebar"] {
    background:
        linear-gradient(
            180deg,
            #0b1020,
            #111827
        );

    border-right: 1px solid rgba(255,255,255,0.08);
}

.sidebar-title {
    font-size: 1.5rem;
    font-weight: 800;
    text-align: center;
}

.sidebar-text {
    color: #94a3b8;
    line-height: 1.6;
    text-align: center;
}

/* ================= FOOTER ================= */

.footer {
    margin-top: 45px;
    text-align: center;
    color: #64748b;
    font-size: 0.85rem;
}

</style>
""", unsafe_allow_html=True)


# ============================================================
# LOAD PICKLE FILES
# ============================================================
@st.cache_resource
def load_models():

    with open("model.pkl", "rb") as f:
        model = pickle.load(f)

    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)

    with open("encoder.pkl", "rb") as f:
        encoder = pickle.load(f)

    return model, scaler, encoder


model, scaler, encoder = load_models()


# ============================================================
# HERO
# ============================================================
st.markdown(
    """
    <div class="hero">

    <div class="hero-title">
        🧠 Personality AI
    </div>

    <div class="hero-subtitle">
        Discover your personality type using Machine Learning
    </div>

    </div>
    """, 
    unsafe_allow_html=True
    )


# ============================================================
# SIDEBAR
# ============================================================
with st.sidebar:

    st.markdown(
        '<div class="sidebar-title">🧠 Personality AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<p class="sidebar-text">'
        'Answer the questions honestly and let the model '
        'analyze your personality patterns.'
        '</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    st.markdown("### 📊 Rating Scale")

    st.markdown("""
    **1** → Very Low

    **5** → Neutral

    **10** → Very High
    """)

    st.markdown("---")

    st.markdown("### 🤖 Model")

    st.write("Logistic Regression")

    st.write("Feature Scaling: Enabled")

    st.write("Target Encoding: LabelEncoder")

    st.markdown("---")

    st.caption("Built with Python + Scikit-learn + Streamlit")


# ============================================================
# INTRODUCTION
# ============================================================
st.markdown("""
<div class="section-header">

<div class="section-title">
✨ Tell us about yourself
</div>

<div class="section-description">
Move each slider according to your personality.
There are no right or wrong answers.
</div>

</div>
""", unsafe_allow_html=True)


# ============================================================
# SOCIAL PERSONALITY
# ============================================================
st.markdown("""
<div class="section-header">

<div class="section-title">
🗣️ Social Personality
</div>

<div class="section-description">
How do you behave around people and social situations?
</div>

</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

with col1:

    social_energy = st.slider(
        "⚡ Social Energy",
        1, 10, 5
    )

    talkativeness = st.slider(
        "💬 Talkativeness",
        1, 10, 5
    )

    group_comfort = st.slider(
        "👥 Group Comfort",
        1, 10, 5
    )

with col2:

    alone_time_preference = st.slider(
        "🌙 Alone Time Preference",
        1, 10, 5
    )

    party_liking = st.slider(
        "🎉 Party Liking",
        1, 10, 5
    )

    listening_skill = st.slider(
        "👂 Listening Skill",
        1, 10, 5
    )

with col3:

    empathy = st.slider(
        "❤️ Empathy",
        1, 10, 5
    )

    friendliness = st.slider(
        "😊 Friendliness",
        1, 10, 5
    )

    public_speaking_comfort = st.slider(
        "🎤 Public Speaking Comfort",
        1, 10, 5
    )


# ============================================================
# THINKING & PERSONALITY
# ============================================================
st.markdown("""
<div class="section-header">

<div class="section-title">
🧩 Thinking & Personality
</div>

<div class="section-description">
How do you think, learn and approach new experiences?
</div>

</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

with col1:

    deep_reflection = st.slider(
        "🔎 Deep Reflection",
        1, 10, 5
    )

    curiosity = st.slider(
        "🔬 Curiosity",
        1, 10, 5
    )

    reading_habit = st.slider(
        "📚 Reading Habit",
        1, 10, 5
    )

with col2:

    gadget_usage = st.slider(
        "📱 Gadget Usage",
        1, 10, 5
    )

    online_social_usage = st.slider(
        "🌐 Online Social Usage",
        1, 10, 5
    )

    risk_taking = st.slider(
        "🎲 Risk Taking",
        1, 10, 5
    )

with col3:

    decision_speed = st.slider(
        "⚡ Decision Speed",
        1, 10, 5
    )

    adventurousness = st.slider(
        "🏔️ Adventurousness",
        1, 10, 5
    )

    excitement_seeking = st.slider(
        "🚀 Excitement Seeking",
        1, 10, 5
    )


# ============================================================
# WORK & ORGANIZATION
# ============================================================
st.markdown("""
<div class="section-header">

<div class="section-title">
💼 Work & Organization
</div>

<div class="section-description">
How do you approach work, planning and responsibilities?
</div>

</div>
""", unsafe_allow_html=True)


col1, col2, col3 = st.columns(3)

with col1:

    organization = st.slider(
        "📋 Organization",
        1, 10, 5
    )

    planning = st.slider(
        "🗓️ Planning",
        1, 10, 5
    )

    routine_preference = st.slider(
        "🔄 Routine Preference",
        1, 10, 5
    )

with col2:

    leadership = st.slider(
        "👑 Leadership",
        1, 10, 5
    )

    work_style_collaborative = st.slider(
        "🤝 Collaborative Work Style",
        1, 10, 5
    )

    spontaneity = st.slider(
        "🎭 Spontaneity",
        1, 10, 5
    )

with col3:

    travel_desire = st.slider(
        "✈️ Travel Desire",
        1, 10, 5
    )

    sports_interest = st.slider(
        "⚽ Sports Interest",
        1, 10, 5
    )


# ============================================================
# PREDICTION BUTTON
# ============================================================
st.markdown("<br>", unsafe_allow_html=True)

predict = st.button(
    "🔮 ANALYZE MY PERSONALITY"
)


# ============================================================
# PREDICTION
# ============================================================
if predict:

    try:

        # ----------------------------------------------------
        # CREATE DATAFRAME
        # ----------------------------------------------------

        input_data = pd.DataFrame([{

            "social_energy": social_energy,

            "alone_time_preference":
                alone_time_preference,

            "talkativeness":
                talkativeness,

            "deep_reflection":
                deep_reflection,

            "group_comfort":
                group_comfort,

            "party_liking":
                party_liking,

            "listening_skill":
                listening_skill,

            "empathy":
                empathy,

            "organization":
                organization,

            "leadership":
                leadership,

            "risk_taking":
                risk_taking,

            "public_speaking_comfort":
                public_speaking_comfort,

            "curiosity":
                curiosity,

            "routine_preference":
                routine_preference,

            "excitement_seeking":
                excitement_seeking,

            "friendliness":
                friendliness,

            "planning":
                planning,

            "spontaneity":
                spontaneity,

            "adventurousness":
                adventurousness,

            "reading_habit":
                reading_habit,

            "sports_interest":
                sports_interest,

            "online_social_usage":
                online_social_usage,

            "travel_desire":
                travel_desire,

            "gadget_usage":
                gadget_usage,

            "work_style_collaborative":
                work_style_collaborative,

            "decision_speed":
                decision_speed
        }])


        # ----------------------------------------------------
        # IMPORTANT:
        # Make sure this order EXACTLY matches X.columns
        # used during training.
        # ----------------------------------------------------

        expected_columns = [
            "social_energy",
            "alone_time_preference",
            "talkativeness",
            "deep_reflection",
            "group_comfort",
            "party_liking",
            "listening_skill",
            "empathy",
            "organization",
            "leadership",
            "risk_taking",
            "public_speaking_comfort",
            "curiosity",
            "routine_preference",
            "excitement_seeking",
            "friendliness",
            "planning",
            "spontaneity",
            "adventurousness",
            "reading_habit",
            "sports_interest",
            "online_social_usage",
            "travel_desire",
            "gadget_usage",
            "work_style_collaborative",
            "decision_speed"
        ]

        input_data = input_data[expected_columns]


        # ----------------------------------------------------
        # SCALE INPUT
        # ----------------------------------------------------

        scaled_input = scaler.transform(input_data)


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        prediction = model.predict(scaled_input)


        # ----------------------------------------------------
        # CONVERT ENCODED LABEL BACK TO ORIGINAL LABEL
        # ----------------------------------------------------

        personality = encoder.inverse_transform(
            prediction
        )[0]


        # ====================================================
        # RESULT
        # ====================================================

        st.markdown(
    f"""
    <div class="result-box">
        <div class="result-small">
            ✨ Your predicted personality type is
        </div>
        <div class="result-big">
            {personality}
        </div>
    </div>
    """,
    unsafe_allow_html=True
    )

        # ====================================================
        # PROBABILITY
        # ====================================================

        if hasattr(model, "predict_proba"):

            probabilities = model.predict_proba(
                scaled_input
            )[0]

            encoded_classes = model.classes_

            personality_classes = encoder.inverse_transform(
                encoded_classes
            )

            probability_df = pd.DataFrame({

                "Personality Type":
                    personality_classes,

                "Probability":
                    probabilities

            })

            probability_df = probability_df.sort_values(
                by="Probability",
                ascending=False
            )

            st.markdown("### 📊 Prediction Confidence")

            st.dataframe(
                probability_df.style.format({
                    "Probability": "{:.2%}"
                }),
                use_container_width=True,
                hide_index=True
            )


        # ====================================================
        # SUCCESS MESSAGE
        # ====================================================

        st.success(
            "✅ Personality analysis completed successfully!"
        )


    except Exception as e:

        st.error(
            "❌ Prediction failed. Please check your model, "
            "scaler and column order."
        )

        st.code(str(e))


# ============================================================
# FOOTER
# ============================================================
st.markdown("""
<div class="footer">

🧠 Personality AI
<br>
Machine Learning • Logistic Regression • Streamlit

</div>
""", unsafe_allow_html=True)