# Spotify User Behavior Analytics Dashboard

An interactive Streamlit dashboard designed to analyze and interpret user listening behavior across multiple Spotify accounts. This project explores how users engage with music and podcasts, uncovering behavioral patterns and translating them into actionable insights.

## Key Insights

- **Exploration vs Repeat Behavior**  
  Understand how frequently users discover new content vs replay existing tracks.

- **Device Usage Patterns**  
  Analyze how listening behavior varies across devices and time of day.

- **Content Preferences**  
  Compare music vs podcast consumption and engagement depth.

- **User Segmentation (K-Means Clustering)**  
  Group users into behavioral segments based on listening habits.

- **Decision-Maker Insights**  
  Translate data into actionable recommendations using an *insight → action → impact* framework.

---

## Tech Stack

- **Python**
- **Pandas**
- **Scikit-learn**
- **Plotly**
- **Streamlit**

---

## Project Structure


spotify-analytics/
├── app.py
├── requirements.txt
├── README.md
└── data/
├── user_1/
│ ├── StreamingHistory_music_0.json
│ ├── StreamingHistory_podcast_0.json
│ └── SearchQueries.json
├── user_2/
├── user_3/
├── user_4/
├── user_5/
└── user_6/


---

## Running Locally

### 1. Install dependencies
```bash
pip install -r requirements.txt

### 2. Add data

Place the data/ folder in the same directory as app.py.

### 3. Run the app
streamlit run app.py

App will open at:
http://localhost:8501

🌐 Deployment (Streamlit Cloud)
Push this repository to GitHub
Go to Streamlit Cloud
Click New App
Select your repo and branch
Set main file to:
app.py
Click Deploy
🔒 Data Note

This project uses Spotify data export files.
For privacy reasons, raw datasets are not included in the public repository.

A sample folder structure is provided to replicate the setup.

📊 App Pages
Page	Description
📋 Overview	Problem statement, KPIs, and analytical approach
📊 Data Summary	Listening patterns, play counts, and distributions
🔍 Exploration vs Repeat	Content discovery vs repeat behavior
📱 Device Usage	Device-based listening patterns
🎙️ Content Preferences	Music vs podcast engagement
🧩 Segmentation	K-Means clustering and user profiles
💡 Insights	Actionable recommendations
🧠 Key Learnings
Behavioral data analysis and feature engineering
Clustering and user segmentation using K-Means
Translating analytics into business insights
Building interactive dashboards with Streamlit
💻 Author

Sara Khandelwal

If you found this interesting feel free to star the repo or connect with me on LinkedIn!
