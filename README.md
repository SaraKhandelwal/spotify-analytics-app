# Spotify User Behavior Analytics — Team 11
## Streamlit Deployment Guide

---

### Project Structure

```
your_repo/
├── app.py
├── requirements.txt
├── README.md               ← this file
└── Phase 2/                ← Spotify data exports (required)
    ├── user_1/
    │   ├── StreamingHistory_music_0.json
    │   ├── StreamingHistory_podcast_0.json
    │   └── SearchQueries.json
    ├── user_2/
    ├── user_3/
    ├── user_4/
    ├── user_5/
    └── user_6/
```

---

### Running Locally

1. **Install Python 3.10+** (if not already installed)

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Place data:** Put the `Phase 2/` folder in the same directory as `app.py`

4. **Launch the app:**
   ```bash
   streamlit run app.py
   ```
   The app will open at `http://localhost:8501`

---

### Deploying to Streamlit Community Cloud (Free)

#### Step 1 — Create a GitHub repository
1. Go to [github.com](https://github.com) → **New Repository**
2. Name it something like `spotify-analytics-team11`
3. Set visibility to **Private** (recommended, since data is personal)

#### Step 2 — Upload your files
Upload these files to the repo root:
- `app.py`
- `requirements.txt`
- The entire `Phase 2/` folder (with all user subfolders)

  Via GitHub web UI: drag-and-drop files, or use Git:
  ```bash
  git init
  git add app.py requirements.txt requirements.txt "Phase 2/"
  git commit -m "Initial commit"
  git remote add origin https://github.com/YOUR_USERNAME/spotify-analytics-team11.git
  git push -u origin main
  ```

#### Step 3 — Deploy on Streamlit Cloud
1. Go to [share.streamlit.io](https://share.streamlit.io)
2. Click **"New app"**
3. Connect your GitHub account
4. Select your repository and branch (`main`)
5. Set **Main file path** to `app.py`
6. Click **Deploy**

The app will be live at:
`https://YOUR_USERNAME-spotify-analytics-team11-app-HASH.streamlit.app`

---

### Notes on Data Privacy

- The `Phase 2/` folder contains personal Spotify data exports.
- Keep the repository **private** if deploying to Streamlit Cloud.
- Only share the deployment URL with course instructors and teammates.
- No data is sent to external services — all processing happens locally.

---

### App Pages

| Page | Content |
|------|---------|
| 📋 Overview | Problem statement, research questions, analytical approach, KPIs |
| 📊 Data Summary | Per-user play counts, listening hours, hourly distribution, raw data preview |
| 🔍 RQ1 · Exploration vs Repeat | Exploration rate chart, engagement comparison, session depth, correlation heatmap |
| 📱 RQ2 · Device Usage Patterns | Session behavior by device, content mix, time-of-day breakdown |
| 🎙️ RQ3 · Content Preferences | Skip rate vs podcast share, listening depth, temporal patterns |
| 🧩 Cluster Segmentation | Unified K-Means segmentation with business labels, feature profiles |
| 💡 Decision-Maker Insights | 6 actionable recommendations with insight → action → impact structure |

---

### Troubleshooting

**"Data directory not found"**
→ Make sure the `Phase 2/` folder is in the same directory as `app.py`.
   The app searches for a folder named exactly `Phase 2`.

**Charts are blank**
→ Check that `StreamingHistory_music_*.json` files exist in at least one user subfolder.

**Slow initial load**
→ The first load processes ~100k records. Streamlit caches all computations —
   subsequent page navigations will be instant.

**Streamlit Cloud memory error**
→ Free tier has 1 GB RAM. With 7 users and ~100k records, this should be fine.
   If you add more users, consider upgrading or reducing the dataset.
