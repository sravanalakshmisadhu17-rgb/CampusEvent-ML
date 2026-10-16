# CampusConnect — Campus and Event Recommendation System

A beginner-friendly machine-learning website built with Python, Flask, Pandas, and Scikit-learn.

## Features
- Student enters interests and selects department, year, and event type.
- TF-IDF vectorization and cosine similarity compare the student profile with event descriptions.
- Results are ranked and shown as website cards with event details and estimated match scores.
- Sample events are stored in `data/events.csv`.

## Run in VS Code (Windows)

1. Extract the ZIP file.
2. Open the extracted `campus_event_recommendation` folder in VS Code.
3. Open **Terminal → New Terminal**.
4. Create a virtual environment:

   ```powershell
   py -m venv .venv
   ```

5. Activate it:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If PowerShell blocks activation, run this once in the current terminal, then activate again:

   ```powershell
   Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
   .\.venv\Scripts\Activate.ps1
   ```

6. Install libraries:

   ```powershell
   python -m pip install -r requirements.txt
   ```

7. Start the website:

   ```powershell
   python app.py
   ```

8. Open this address in your browser: **http://127.0.0.1:5000**

Keep the terminal running while using the website. Press `Ctrl+C` in the terminal to stop it.

## How the ML works
- `TfidfVectorizer` converts the event text and student profile into numerical vectors.
- `cosine_similarity` scores how closely each event matches the profile.
- Selected department, year, and event type provide additional ranking boosts.
- The highest-scoring events appear first.

## Important project note
The CSV contains demonstration events and dates. Replace them with verified campus event information before presenting this as a live campus service. The displayed match percentage is a ranking score, not a probability that a student will attend.
