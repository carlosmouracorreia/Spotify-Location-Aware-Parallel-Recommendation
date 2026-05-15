import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset


def round_coords(lat, lon, precision=2):
    """Round latitude and longitude to create region features."""
    return f"region_lat_{round(lat, precision)}_lon_{round(lon, precision)}"


def load_top_songs_dataset(filepath):
    """Load and preprocess the Spotify songs dataset."""
    raw_songs_df = pd.read_csv(filepath)

    songs_df = raw_songs_df.rename(columns={
        'Track': 'name',
        'Artist': 'artist',
        'Danceability': 'danceability',
        'Energy': 'energy',
        'Valence': 'valence',
        'Time_Signature': 'tempo',
        'Genre': 'genre'
    })

    songs_df['id'] = ['song' + str(i + 1) for i in range(len(songs_df))]
    songs_df = songs_df[['id', 'name', 'artist', 'danceability', 'energy', 'valence', 'tempo', 'genre']]

    return songs_df


# Initialize dataset and model
DATA_FILE_PATH = Path(os.getenv('DATA_FILE_PATH', Path(__file__).resolve().parent / 'data' / 'ClassicHit15k.csv'))
songs_df = load_top_songs_dataset(DATA_FILE_PATH)

# Base user definitions for initial model training
_base_users = {
    'alice': {'lat': 40.7128, 'lon': -74.0060, 'city': 'NYC'},
    'bob': {'lat': 40.7135, 'lon': -74.0050, 'city': 'NYC'},
    'carol': {'lat': 34.0522, 'lon': -118.2437, 'city': 'LA'},
}

# Base interactions for initial model training
_base_interactions = [
    ('alice', songs_df.iloc[0]['id']),
    ('alice', songs_df.iloc[1]['id']),
    ('bob', songs_df.iloc[0]['id']),
    ('bob', songs_df.iloc[2]['id']),
    ('carol', songs_df.iloc[3]['id']),
    ('carol', songs_df.iloc[4]['id']),
]

interactions_df = pd.DataFrame(_base_interactions, columns=['user_id', 'song_id'])

# Initialize song_plays list for tracking
song_plays = []

# Build LightFM dataset
dataset = Dataset()
dataset.fit(users=_base_users.keys(), items=songs_df['id'])

# Create user features
user_feature_list = [(uid, [round_coords(data['lat'], data['lon'])]) for uid, data in _base_users.items()]

# Create item features
item_feature_list = []
for _, row in songs_df.iterrows():
    features = [f"valence:{round(row['valence'], 2)}", 
                f"tempo:{int(row['tempo'])}", 
                f"danceability:{round(row['danceability'], 2)}"]
    item_feature_list.append((row['id'], features))

# Fit feature vocab
dataset.fit_partial(
    users=_base_users.keys(),
    items=songs_df['id'],
    user_features=[f for _, ftrs in user_feature_list for f in ftrs],
    item_features=[f for _, ftrs in item_feature_list for f in ftrs],
)

# Build interaction matrix
interaction_tuples = list(zip(interactions_df['user_id'], interactions_df['song_id']))
interactions_matrix, _ = dataset.build_interactions(interaction_tuples)

# Build feature matrices
user_features_matrix = dataset.build_user_features(user_feature_list)
item_features_matrix = dataset.build_item_features(item_feature_list)

# Train the model
model = LightFM(loss='logistic')
model.fit(interactions_matrix,
          user_features=user_features_matrix,
          item_features=item_features_matrix,
          epochs=30,
          num_threads=2)


def trail_recommendation(user_id, model, dataset, user_features_matrix, item_features_matrix, interactions_df, song_plays, top_n=3):
    """Generate top N song recommendations for a user."""
    user_idx = dataset.mapping()[0][user_id]
    song_idx_to_id = {v: k for k, v in dataset.mapping()[2].items()}

    scores = model.predict(user_idx,
                           np.arange(len(song_idx_to_id)),
                           user_features=user_features_matrix,
                           item_features=item_features_matrix)

    # Filter out already played songs
    already_played = interactions_df[interactions_df['user_id'] == user_id]['song_id'].tolist()
    already_played_idx = [dataset.mapping()[2][sid] for sid in already_played if sid in dataset.mapping()[2]]
    scores[already_played_idx] = -np.inf

    top_items = np.argsort(-scores)[:top_n]
    trail = []

    # Fetch recommendations with metadata
    recent_plays = {row['song']: row for row in song_plays}
    for i in top_items:
        song_id = song_idx_to_id[i]
        meta = recent_plays.get(song_id, {"user": None, "city": None, "region": None})
        trail.append({
            "song": song_id,
            "song_name": songs_df[songs_df['id'] == song_id]['name'].values[0],
            "artist": songs_df[songs_df['id'] == song_id]['artist'].values[0],
            "last_user_playing": meta["user"],
            "city": meta["city"],
            "region": meta["region"],
        })
    return trail
