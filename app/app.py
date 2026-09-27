import streamlit as st
import numpy as np
import pandas as pd
import joblib

st.set_page_config(page_title="Social Media Fraud Detector", page_icon="🛡️", layout="centered")

@st.cache_resource
def load_model_artifacts():
    # Loads the trained classical/tree pipeline
    model = joblib.load("models/best_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_columns.pkl")
    return model, scaler, feature_cols

model, scaler, feature_columns = load_model_artifacts()

st.title("🛡️ Behavioral Social Media Fraud Detector")
st.write("Analyze account metrics in real time to assess fraud or bot behavior.")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.subheader("Profile Demographics")
    followers = st.number_input("Followers Count", min_value=0, value=85, step=1)
    following = st.number_input("Following Count", min_value=0, value=950, step=1)
    account_age = st.number_input("Account Age (Days)", min_value=1, value=20, step=1)
    has_pic = st.selectbox("Has Profile Picture?", ["Yes", "No"])

with col2:
    st.subheader("Behavioral Signals")
    posts_day = st.number_input("Avg Posts Per Day", min_value=0.0, value=25.0, step=1.0)
    hashtags = st.number_input("Avg Hashtags Per Post", min_value=0.0, value=5.5, step=0.5)
    urls = st.number_input("Avg URLs Per Post", min_value=0.0, value=2.0, step=0.5)

st.markdown("---")

if st.button("Evaluate Profile Risk", use_container_width=True):
    # Compute behavioral features
    pic_flag = 1 if has_pic == "Yes" else 0
    ratio = followers / (following + 1)
    posting_rate = posts_day / (account_age + 1)
    spam_intensity = hashtags * urls
    
    input_data = pd.DataFrame([{
        'followers_count': followers,
        'following_count': following,
        'posts_per_day': posts_day,
        'avg_hashtags_per_post': hashtags,
        'avg_urls_per_post': urls,
        'account_age_days': account_age,
        'has_profile_pic': pic_flag,
        'follower_following_ratio': ratio,
        'posting_rate_per_account_day': posting_rate,
        'spam_content_intensity': spam_intensity
    }])[feature_columns]
    
    # Scale and predict
    input_scaled = scaler.transform(input_data)
    fraud_probability = model.predict_proba(input_scaled)[0][1]
    
    st.subheader("Analysis Verdict")
    if fraud_probability >= 0.5:
        st.error(f"🚨 **High Risk Account (Potential Fraud / Bot)**\n\nEstimated Risk Confidence: **{fraud_probability * 100:.2f}%**")
    else:
        st.success(f"✅ **Low Risk Account (Legitimate User)**\n\nEstimated Legitimate Confidence: **{(1 - fraud_probability) * 100:.2f}%**")
        
    st.markdown("#### Key Calculated Ratios")
    st.write(f"- **Follower-to-Following Ratio:** `{ratio:.4f}`")
    st.write(f"- **Posting Rate per Lifetime Day:** `{posting_rate:.4f}`")
    st.write(f"- **Spam Content Intensity Index:** `{spam_intensity:.2f}`")