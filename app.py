from flask import Flask, render_template, request, jsonify
from recommender import get_recommendations, get_all_titles

app = Flask(__name__)

@app.route('/')
def index():
    titles = get_all_titles()
    return render_template('index.html', titles=titles)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    movie = data.get('movie', '')
    matched, results = get_recommendations(movie)
    if not results:
        return jsonify({'error': f'Movie not found: {movie}'}), 404
    return jsonify({'matched': matched, 'recommendations': results})

if __name__ == '__main__':
    app.run(debug=True)