import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ── Dataset (100 movies built in) ──
MOVIES = [
    {"id":1,"title":"The Dark Knight","genres":"Action Crime Thriller","rating":9.0},
    {"id":2,"title":"Inception","genres":"Action Sci-Fi Thriller","rating":8.8},
    {"id":3,"title":"Interstellar","genres":"Adventure Drama Sci-Fi","rating":8.6},
    {"id":4,"title":"The Matrix","genres":"Action Sci-Fi","rating":8.7},
    {"id":5,"title":"Pulp Fiction","genres":"Crime Drama","rating":8.9},
    {"id":6,"title":"The Godfather","genres":"Crime Drama","rating":9.2},
    {"id":7,"title":"Fight Club","genres":"Drama Thriller","rating":8.8},
    {"id":8,"title":"Forrest Gump","genres":"Drama Romance","rating":8.8},
    {"id":9,"title":"The Shawshank Redemption","genres":"Drama","rating":9.3},
    {"id":10,"title":"Goodfellas","genres":"Biography Crime Drama","rating":8.7},
    {"id":11,"title":"The Silence of the Lambs","genres":"Crime Horror Thriller","rating":8.6},
    {"id":12,"title":"Schindler's List","genres":"Biography Drama History","rating":9.0},
    {"id":13,"title":"The Lord of the Rings","genres":"Adventure Drama Fantasy","rating":8.9},
    {"id":14,"title":"Star Wars A New Hope","genres":"Action Adventure Sci-Fi","rating":8.6},
    {"id":15,"title":"Avengers Endgame","genres":"Action Adventure Sci-Fi","rating":8.4},
    {"id":16,"title":"Titanic","genres":"Drama Romance","rating":7.9},
    {"id":17,"title":"Jurassic Park","genres":"Adventure Sci-Fi Thriller","rating":8.2},
    {"id":18,"title":"The Lion King","genres":"Animation Adventure Drama","rating":8.5},
    {"id":19,"title":"Toy Story","genres":"Animation Adventure Comedy","rating":8.3},
    {"id":20,"title":"Finding Nemo","genres":"Animation Adventure Comedy","rating":8.2},
    {"id":21,"title":"Up","genres":"Animation Adventure Drama","rating":8.3},
    {"id":22,"title":"WALL-E","genres":"Animation Drama Romance Sci-Fi","rating":8.4},
    {"id":23,"title":"The Incredibles","genres":"Animation Action Adventure","rating":8.0},
    {"id":24,"title":"Spirited Away","genres":"Animation Adventure Family","rating":8.6},
    {"id":25,"title":"Princess Mononoke","genres":"Animation Action Adventure","rating":8.4},
    {"id":26,"title":"Parasite","genres":"Comedy Drama Thriller","rating":8.5},
    {"id":27,"title":"Oldboy","genres":"Action Drama Mystery Thriller","rating":8.4},
    {"id":28,"title":"City of God","genres":"Crime Drama","rating":8.6},
    {"id":29,"title":"Pan's Labyrinth","genres":"Drama Fantasy","rating":8.2},
    {"id":30,"title":"Amelie","genres":"Comedy Romance","rating":8.3},
    {"id":31,"title":"Get Out","genres":"Horror Mystery Thriller","rating":7.7},
    {"id":32,"title":"Us","genres":"Horror Mystery Thriller","rating":6.8},
    {"id":33,"title":"A Quiet Place","genres":"Drama Horror Sci-Fi","rating":7.5},
    {"id":34,"title":"Hereditary","genres":"Drama Horror Mystery","rating":7.3},
    {"id":35,"title":"The Conjuring","genres":"Horror Mystery Thriller","rating":7.5},
    {"id":36,"title":"It","genres":"Horror","rating":7.3},
    {"id":37,"title":"The Shining","genres":"Drama Horror","rating":8.4},
    {"id":38,"title":"Alien","genres":"Horror Sci-Fi","rating":8.5},
    {"id":39,"title":"Aliens","genres":"Action Horror Sci-Fi","rating":8.3},
    {"id":40,"title":"Blade Runner","genres":"Drama Sci-Fi","rating":8.1},
    {"id":41,"title":"Blade Runner 2049","genres":"Drama Mystery Sci-Fi","rating":8.0},
    {"id":42,"title":"Ex Machina","genres":"Drama Sci-Fi Thriller","rating":7.7},
    {"id":43,"title":"Her","genres":"Drama Romance Sci-Fi","rating":8.0},
    {"id":44,"title":"2001 A Space Odyssey","genres":"Adventure Mystery Sci-Fi","rating":8.3},
    {"id":45,"title":"Gravity","genres":"Adventure Drama Sci-Fi","rating":7.7},
    {"id":46,"title":"The Martian","genres":"Adventure Drama Sci-Fi","rating":8.0},
    {"id":47,"title":"Apollo 13","genres":"Adventure Drama History","rating":7.6},
    {"id":48,"title":"Whiplash","genres":"Drama Music","rating":8.5},
    {"id":49,"title":"Black Swan","genres":"Drama Thriller","rating":8.0},
    {"id":50,"title":"La La Land","genres":"Comedy Drama Music Romance","rating":8.0},
    {"id":51,"title":"The Grand Budapest Hotel","genres":"Adventure Comedy Crime","rating":8.1},
    {"id":52,"title":"Moonrise Kingdom","genres":"Adventure Comedy Drama Romance","rating":7.8},
    {"id":53,"title":"Knives Out","genres":"Comedy Crime Drama Mystery","rating":7.9},
    {"id":54,"title":"Gone Girl","genres":"Drama Mystery Thriller","rating":8.1},
    {"id":55,"title":"Se7en","genres":"Crime Drama Mystery Thriller","rating":8.6},
    {"id":56,"title":"Zodiac","genres":"Crime Drama Mystery Thriller","rating":7.7},
    {"id":57,"title":"Prisoners","genres":"Crime Drama Mystery Thriller","rating":8.1},
    {"id":58,"title":"Arrival","genres":"Drama Mystery Sci-Fi","rating":7.9},
    {"id":59,"title":"Edge of Tomorrow","genres":"Action Sci-Fi","rating":7.9},
    {"id":60,"title":"Mad Max Fury Road","genres":"Action Adventure Sci-Fi","rating":8.1},
    {"id":61,"title":"John Wick","genres":"Action Crime Thriller","rating":7.4},
    {"id":62,"title":"Mission Impossible Fallout","genres":"Action Adventure Thriller","rating":7.7},
    {"id":63,"title":"Casino Royale","genres":"Action Adventure Thriller","rating":8.0},
    {"id":64,"title":"No Country for Old Men","genres":"Crime Drama Thriller","rating":8.2},
    {"id":65,"title":"There Will Be Blood","genres":"Drama","rating":8.2},
    {"id":66,"title":"12 Years a Slave","genres":"Biography Drama History","rating":8.1},
    {"id":67,"title":"The Revenant","genres":"Adventure Drama","rating":8.0},
    {"id":68,"title":"Braveheart","genres":"Biography Drama History","rating":8.3},
    {"id":69,"title":"Gladiator","genres":"Action Adventure Drama","rating":8.5},
    {"id":70,"title":"300","genres":"Action Drama History","rating":7.6},
    {"id":71,"title":"Troy","genres":"Action Adventure Drama","rating":7.2},
    {"id":72,"title":"Kingdom of Heaven","genres":"Action Adventure Drama","rating":7.2},
    {"id":73,"title":"The Last Samurai","genres":"Action Drama History","rating":7.7},
    {"id":74,"title":"Kill Bill Vol 1","genres":"Action Crime Thriller","rating":8.1},
    {"id":75,"title":"Django Unchained","genres":"Drama Western","rating":8.4},
    {"id":76,"title":"Inglourious Basterds","genres":"Adventure Drama War","rating":8.3},
    {"id":77,"title":"The Hateful Eight","genres":"Crime Drama Mystery Western","rating":7.8},
    {"id":78,"title":"Once Upon a Time in Hollywood","genres":"Comedy Drama","rating":7.6},
    {"id":79,"title":"Joker","genres":"Crime Drama Thriller","rating":8.4},
    {"id":80,"title":"Logan","genres":"Action Drama Sci-Fi","rating":8.1},
    {"id":81,"title":"Spider-Man Into the Spider-Verse","genres":"Animation Action Adventure","rating":8.4},
    {"id":82,"title":"Black Panther","genres":"Action Adventure Sci-Fi","rating":7.3},
    {"id":83,"title":"Iron Man","genres":"Action Adventure Sci-Fi","rating":7.9},
    {"id":84,"title":"Captain America Civil War","genres":"Action Adventure Sci-Fi","rating":7.8},
    {"id":85,"title":"Thor Ragnarok","genres":"Action Adventure Comedy Sci-Fi","rating":7.9},
    {"id":86,"title":"Guardians of the Galaxy","genres":"Action Adventure Comedy Sci-Fi","rating":8.0},
    {"id":87,"title":"Doctor Strange","genres":"Action Adventure Fantasy Sci-Fi","rating":7.5},
    {"id":88,"title":"The Social Network","genres":"Biography Drama","rating":7.7},
    {"id":89,"title":"Moneyball","genres":"Biography Drama Sport","rating":7.6},
    {"id":90,"title":"The Big Short","genres":"Biography Comedy Drama","rating":7.8},
    {"id":91,"title":"Wolf of Wall Street","genres":"Biography Comedy Crime Drama","rating":8.2},
    {"id":92,"title":"Catch Me If You Can","genres":"Biography Crime Drama","rating":8.1},
    {"id":93,"title":"The Truman Show","genres":"Comedy Drama Sci-Fi","rating":8.2},
    {"id":94,"title":"Eternal Sunshine of the Spotless Mind","genres":"Drama Romance Sci-Fi","rating":8.3},
    {"id":95,"title":"500 Days of Summer","genres":"Comedy Drama Romance","rating":7.7},
    {"id":96,"title":"About Time","genres":"Comedy Drama Fantasy Romance","rating":7.8},
    {"id":97,"title":"The Notebook","genres":"Drama Romance","rating":7.8},
    {"id":98,"title":"Pride and Prejudice","genres":"Drama Romance","rating":7.8},
    {"id":99,"title":"Coco","genres":"Animation Adventure Family Music","rating":8.4},
    {"id":100,"title":"Soul","genres":"Animation Adventure Comedy Drama Music","rating":8.1},
]

df = pd.DataFrame(MOVIES)

# ── Content-based filtering (genre similarity) ──
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df['genres'])
content_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)

# ── Collaborative filtering (rating similarity) ──
rating_matrix = df[['rating']].values
collab_sim = cosine_similarity(rating_matrix, rating_matrix)

# ── Combined similarity ──
combined_sim = 0.7 * content_sim + 0.3 * collab_sim

def get_recommendations(movie_title, n=6):
    matches = df[df['title'].str.lower() == movie_title.lower()]
    if matches.empty:
        # fuzzy fallback
        matches = df[df['title'].str.lower().str.contains(movie_title.lower())]
    if matches.empty:
        return None, []
    idx = matches.index[0]
    matched_title = df.loc[idx, 'title']
    scores = list(enumerate(combined_sim[idx]))
    scores = sorted(scores, key=lambda x: x[1], reverse=True)
    scores = [s for s in scores if s[0] != idx][:n]
    results = []
    for i, score in scores:
        results.append({
            "title": df.loc[i, 'title'],
            "genres": df.loc[i, 'genres'].replace(' ', ', '),
            "rating": df.loc[i, 'rating'],
            "score": round(score * 100, 1)
        })
    return matched_title, results

def get_all_titles():
    return sorted(df['title'].tolist())