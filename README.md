# 🎮 GameFinder AI — Hybrid Video Game Recommendation Engine

An interactive machine learning recommendation engine built with **Streamlit**, **Scikit-learn**, **Pandas**, and **Plotly** that analyzes 16,000+ video game titles across platforms, genres, publishers, and global sales metrics to deliver personalized game discovery.

---

## ✨ Key Features

- **🎯 Content-Based Filtering:** Computes **Cosine**, **Euclidean**, and **Hybrid (70% Cosine + 30% Euclidean)** similarity matrices over scaled numerical features and one-hot encoded categorical attributes (`Genre`, `Platform`, `Publisher`) with multi-platform duplicate filtering.
- **💝 Personalized Multi-Game Engine:** Aggregates similarity vectors across a user's liked titles while applying preference multipliers for favorite genres/platforms and penalizing disliked games.
- **🔥 Popularity & Trend Discovery:** Ranks titles using a custom composite **Popularity Score** factoring global sales, release recency, and multi-platform availability with dynamic year, genre, platform, and publisher filters.
- **🔍 Hidden Gems Finder:** Surfaces high-potential, mid-tier titles across diverse genres outside mainstream blockbusters.
- **📊 Interactive Analytics Dashboard:** Visualizes top genres, platform market share, and historical global sales trajectories using **Plotly**, with one-click CSV export for all recommendation lists.

---

## 🛠️ Tech Stack

- **Frontend & App Framework:** Streamlit
- **Machine Learning & Preprocessing:** Scikit-learn (`StandardScaler`, `OneHotEncoder`, `MinMaxScaler`, `cosine_similarity`, `euclidean_distances`)
- **Data Manipulation:** Pandas, NumPy
- **Visualization:** Plotly Express & Graph Objects

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone [https://github.com/itsmeusaid/gamefinder-ai.git](https://github.com/itsmeusaid/gamefinder-ai.git)
cd gamefinder-ai
