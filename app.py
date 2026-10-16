from flask import Flask, render_template, request
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path

app = Flask(__name__)
DATA_PATH = Path(__file__).parent / "data" / "events.csv"

def load_events():
    return pd.read_csv(DATA_PATH).fillna("")

def recommend_events(interests, department, year, event_type, top_n=6):
    events = load_events()
    interests = interests.strip()

    # Combine event details into text that the ML model can compare.
    event_text = (
        events["title"].astype(str) + " " +
        events["category"].astype(str) + " " +
        events["description"].astype(str) + " " +
        events["skills"].astype(str) + " " +
        events["department"].astype(str)
    )

    # Include the student's interests and preferences in their profile.
    profile = f"{interests} {department} {event_type}"

    vectorizer = TfidfVectorizer(stop_words="english")
    matrix = vectorizer.fit_transform(list(event_text) + [profile])
    scores = cosine_similarity(matrix[-1], matrix[:-1]).flatten()

    events["score"] = scores

    # Small preference boosts make the results more tailored.
    if department and department != "Any":
        events.loc[events["department"].str.contains(department, case=False, na=False), "score"] += 0.12
    if event_type and event_type != "Any":
        events.loc[events["category"].str.contains(event_type, case=False, na=False), "score"] += 0.18
    if year and year != "Any":
        events.loc[events["eligible_years"].str.contains(year, case=False, na=False), "score"] += 0.08

    results = events.sort_values("score", ascending=False).head(top_n).copy()
    results["match_percent"] = (results["score"].clip(0, 1) * 100).round().astype(int)
    return results.to_dict(orient="records")

@app.route("/", methods=["GET", "POST"])
def home():
    recommendations = []
    submitted = False
    form_data = {
        "interests": "",
        "department": "Any",
        "year": "Any",
        "event_type": "Any",
    }

    if request.method == "POST":
        submitted = True
        form_data = {
            "interests": request.form.get("interests", ""),
            "department": request.form.get("department", "Any"),
            "year": request.form.get("year", "Any"),
            "event_type": request.form.get("event_type", "Any"),
        }
        if form_data["interests"].strip():
            recommendations = recommend_events(**form_data)

    return render_template(
        "index.html",
        recommendations=recommendations,
        submitted=submitted,
        form_data=form_data,
    )

if __name__ == "__main__":
    app.run(debug=True)
