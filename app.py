from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np
import requests
import pandas as pd
import os
from dotenv import load_dotenv
from huggingface_hub import hf_hub_download

load_dotenv()

app = Flask(__name__)

HF_REPO = "Prifea/hit-predictor"

def load_model_files():
    model_path = 'hit_model_v3.pkl'
    le_genre_path = 'le_genre_v3.pkl'
    le_explicit_path = 'le_explicit_genre_v3.pkl'

    if not os.path.exists(model_path):
        model_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="hit_model_v3.pkl", #
        )
    if not os.path.exists(le_genre_path):
        le_genre_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="le_genre_v3.pkl", 
        )
    if not os.path.exists(le_explicit_path):
        le_explicit_path = hf_hub_download(
            repo_id=HF_REPO,
            filename="le_explicit_genre_v3.pkl", 
        )

    with open(model_path, 'rb') as f:
        model = pickle.load(f)
    with open(le_genre_path, 'rb') as f:
        le_genre = pickle.load(f)
    with open(le_explicit_path, 'rb') as f:
        le_explicit_genre = pickle.load(f)

    return model, le_genre, le_explicit_genre

print("Loading model files...")
model, le_genre, le_explicit_genre = load_model_files()
print("Model loaded successfully!")

TIER_EXPLANATIONS = {
    'Viral': 'This song has the audio profile of the top 1% of Spotify tracks. It shows strong patterns matching chart-dominating hits.',
    'Hit':   'This song has the profile of a well-performing track. It is likely to build a solid fanbase and feature in playlists.',
    'Mid':   'This song shows moderate popularity potential. It may perform well within a specific genre or regional market.',
    'Flop':  'Based on its audio features, this song may struggle to find a wide audience on Spotify.'
}


def get_artist_image(artist_name):
    try:
        url = 'https://api.deezer.com/search/artist'
        params = {'q': artist_name, 'limit': 1}
        response = requests.get(url, params=params, timeout=5)
        data = response.json()
        artists = data.get('data', [])
        if artists:
            picture = artists[0].get('picture_xl') or \
                      artists[0].get('picture_big') or \
                      artists[0].get('picture')
            return picture
        return None
    except Exception:
        return None


@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'GET':
        return render_template('predict.html')

    try:
        duration_minutes = float(request.form['duration'])
        danceability     = float(request.form['danceability']) / 100
        energy           = float(request.form['energy']) / 100
        loudness         = float(request.form['loudness'])
        speechiness      = float(request.form['speechiness']) / 100
        acousticness     = float(request.form['acousticness']) / 100
        instrumentalness = float(request.form['instrumentalness']) / 100
        tempo            = float(request.form['tempo'])
        genre_str        = request.form['genre']
        artist_name      = request.form.get('artist', '').strip()

        explicit_str       = 'explicit' if request.form['explicit_genre'] == '1' else 'clean'
        explicit_genre_str = f"{genre_str}_{explicit_str}"
        if explicit_genre_str not in le_explicit_genre.classes_:
            explicit_genre_str = f"{genre_str}_clean"

        genre_encoded          = le_genre.transform([genre_str])[0]
        explicit_genre_encoded = le_explicit_genre.transform([explicit_genre_str])[0]

        features = pd.DataFrame([[
            duration_minutes, danceability, energy, loudness,
            speechiness, acousticness, instrumentalness, tempo,
            genre_encoded, explicit_genre_encoded
        ]], columns=[
            'duration_minutes', 'danceability', 'energy', 'loudness',
            'speechiness', 'acousticness', 'instrumentalness', 'tempo',
            'track_genre', 'explicit_genre'
        ])

        prediction    = model.predict(features)[0]
        probabilities = model.predict_proba(features)[0]

        prob_dict = {
            'flop':  round(float(probabilities[0]), 3),
            'hit':   round(float(probabilities[1]), 3),
            'mid':   round(float(probabilities[2]), 3),
            'viral': round(float(probabilities[3]), 3),
        }

        artist_image = None
        if artist_name:
            artist_image = get_artist_image(artist_name)

        return jsonify({
            'prediction':    prediction,
            'tier':          prediction.lower(),
            'explanation':   TIER_EXPLANATIONS.get(prediction, ''),
            'probabilities': prob_dict,
            'artist_image':  artist_image,
            'artist_name':   artist_name
        })

    except ValueError as e:
        return jsonify({'error': f'Invalid value: {str(e)}'}), 400
    except KeyError as e:
        return jsonify({'error': f'Missing field: {str(e)}'}), 400
    except Exception as e:
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)