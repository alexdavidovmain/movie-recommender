🎬 Movie Recommender System
What is this?
Imagine you just watched Inception and loved it. You want something similar but don't know what to watch next. You could ask a friend who has seen every movie ever made and knows exactly what you'd like based on the genres and style you enjoy.
This project is that friend. You type in a movie you love, and it instantly recommends 6 movies you'll probably enjoy too — using real math and machine learning to figure out what's similar.

How does it work?
It uses two methods combined together:
Method 1 — Content-based filtering
Looks at the genres of the movie you picked and finds other movies with similar genres. If you like action sci-fi thrillers, it finds more action sci-fi thrillers.
Method 2 — Collaborative filtering
Looks at ratings. Movies that are rated similarly tend to appeal to similar people. So if you liked a 9.0 rated movie, it leans toward other highly rated films.
Both scores are combined — 70% genre similarity + 30% rating similarity — to give the final recommendations.

What I built it with
ToolWhat it's forPythonThe programming language everything is written inpandasHandling the movie datasetscikit-learnTF-IDF vectorizer and cosine similarity for finding similar moviesFlaskThe web framework that runs the browser interfaceHTML/CSS/JSThe visual interface you interact with

How I built it — step by step
Step 1 — Built the dataset
Created a dataset of 100 well-known movies, each with a title, genres, and IMDb rating. No external API needed — it's all built in.
Step 2 — Content-based similarity
Used TF-IDF (Term Frequency-Inverse Document Frequency) to turn each movie's genres into a mathematical vector. Then used cosine similarity to measure how close any two movies are to each other based on genres.
Step 3 — Collaborative similarity
Used cosine similarity on the ratings column. Movies with similar ratings cluster together, which helps surface quality films in the same tier.
Step 4 — Combined the two
Weighted the two similarity scores together: 70% content + 30% collaborative. This gives genre-accurate recommendations that also tend to be highly rated.
Step 5 — Built the web interface
Used Flask to serve the app in the browser. The frontend has a search input with live autocomplete, and displays the 6 recommendations as cards showing title, genres, rating, and match percentage.

How to run it
Install dependencies:
pip install pandas scikit-learn flask
Run the app:
python app.py
Open in browser:
http://127.0.0.1:5000
Type any movie name and hit Enter or click Get Recs.

What I learned

How TF-IDF works and why it's useful for text similarity
The difference between content-based and collaborative filtering
How to combine multiple signals into one recommendation score
How to build and serve a web app with Flask
Connecting a Python backend to a browser frontend


Built by Alexander Davidov
