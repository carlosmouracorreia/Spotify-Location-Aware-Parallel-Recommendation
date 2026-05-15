import os
from pathlib import Path

import numpy as np
import pandas as pd
from lightfm import LightFM
from lightfm.data import Dataset
import random
from datetime import datetime, timedelta

# round the coordinates to a certain precision - check this in the future
def round_coords(lat, lon, precision=2):
    return f"region_lat_{round(lat, precision)}_lon_{round(lon, precision)}"

# Load the Kaggle dataset
def load_top_songs_dataset(filepath):
    # Load the dataset
    raw_songs_df = pd.read_csv(filepath)

    # Preprocess the dataset to match the structure of songs_df
    songs_df = raw_songs_df.rename(columns={
        'Track': 'name',
        'Artist': 'artist',
        'Danceability': 'danceability',
        'Energy': 'energy',
        'Valence': 'valence',
        'Time_Signature': 'tempo',
        'Genre': 'genre'
    })

    # Generate unique IDs for each song
    songs_df['id'] = ['song' + str(i + 1) for i in range(len(songs_df))]

    # Keep only the necessary columns
    songs_df = songs_df[['id', 'name', 'artist', 'danceability', 'energy', 'valence', 'tempo', 'genre']]

    return songs_df

DATA_FILE_PATH = Path(os.getenv('DATA_FILE_PATH', Path(__file__).resolve().parent / 'data' / 'ClassicHit15k.csv'))

# Load the dataset
songs_df = load_top_songs_dataset(DATA_FILE_PATH)

# User definitions
users = {
    'alice': {'lat': 40.7128, 'lon': -74.0060, 'city': 'NYC'},  # NYC
    'bob': {'lat': 40.7135, 'lon': -74.0050, 'city': 'NYC'},    # also NYC, close
    'carol': {'lat': 34.0522, 'lon': -118.2437, 'city': 'LA'},  # LA
}

# Generate interactions (some users have heard certain songs)
# put songs here that you want to use/know to improve efficiency...
interactions = [
    ('alice', songs_df.iloc[0]['id']),
    ('alice', songs_df.iloc[1]['id']),
    ('bob', songs_df.iloc[0]['id']),
    ('bob', songs_df.iloc[2]['id']),
    ('carol', songs_df.iloc[3]['id']),
    ('carol', songs_df.iloc[4]['id']),
]

interactions_df = pd.DataFrame(interactions, columns=['user_id', 'song_id'])

# Generate randomized song plays with timestamp, region, and city
song_plays = []
base_time = datetime(2025, 4, 10, 12, 0, 0)

for user, song in interactions_df[['user_id', 'song_id']].values:
    user_data = users[user]
    region = round_coords(user_data['lat'], user_data['lon'])
    city = user_data['city']
    
    # Generate a fake recent timestamp with slight variation
    random_offset = timedelta(hours=random.randint(0, 5), minutes=random.randint(0, 59))
    timestamp = base_time + random_offset

    song_plays.append({
        'song': song,
        'user': user,
        'region': region,
        'city': city,
        'timestamp': timestamp
    })

# Generate user features
user_features = {
    uid: {
        'region': round_coords(data['lat'], data['lon'])
    } for uid, data in users.items()
}

# Build dataset and features for LightFM
dataset = Dataset()
dataset.fit(users=users.keys(), items=songs_df['id'])

# User features
user_feature_list = [(uid, [f['region']]) for uid, f in user_features.items()]

# Song features
item_feature_list = []
for _, row in songs_df.iterrows():
    features = [f"valence:{round(row['valence'], 2)}", 
                f"tempo:{int(row['tempo'])}", 
                f"danceability:{round(row['danceability'], 2)}"]
    item_feature_list.append((row['id'], features))

# Fit feature vocab
dataset.fit_partial(
    users=users.keys(),
    items=songs_df['id'],
    user_features=[f for _, ftrs in user_feature_list for f in ftrs],
    item_features=[f for _, ftrs in item_feature_list for f in ftrs],
)

# Build interactions
interaction_tuples = list(zip(interactions_df['user_id'], interactions_df['song_id']))

(interactions_matrix, weights) = dataset.build_interactions(interaction_tuples)

# Build user/item feature matrices
user_features_matrix = dataset.build_user_features(user_feature_list)
item_features_matrix = dataset.build_item_features(item_feature_list)

# Train the model
model = LightFM(loss='logistic')
model.fit(interactions_matrix,
            user_features=user_features_matrix,
            item_features=item_features_matrix,
            epochs=30,
            num_threads=2)

def main():


    # Add a new song to the model
    new_song = {
        "id": "song16",
        "name": "Dancing Queen",
        "artist": "ABBE",
        "danceability": 0.9,
        "energy": 0.8,
        "valence": 0.7,
        "tempo": 120
    }

    # Format item features
    song_features = [f"valence:{round(new_song['valence'], 2)}", 
                     f"tempo:{int(new_song['tempo'])}", 
                     f"danceability:{round(new_song['danceability'], 2)}"]

    # Add new song to model
    dataset.fit_partial(items=[new_song['id']], item_features=song_features)
    item_feature_list.append((new_song['id'], song_features))

    item_features_matrix_new = dataset.build_item_features(item_feature_list)

    # Refit the model with the updated item features
    model.fit(
        interactions_matrix, 
        item_features=item_features_matrix_new, 
        user_features=user_features_matrix,
        epochs=30, 
        num_threads=2
    )



    # Add a new song to the model
    new_song = {
        "id": "song16",
        "name": "Dancing Queen",
        "artist": "ABBA",
        "danceability": 0.9,
        "energy": 0.8,
        "valence": 0.7,
        "tempo": 120
    }

    # Format item features
    song_features = [f"valence:{round(new_song['valence'], 2)}", 
                     f"tempo:{int(new_song['tempo'])}", 
                     f"danceability:{round(new_song['danceability'], 2)}"]

    # Add new song to model
    dataset.fit_partial(items=[new_song['id']], item_features=song_features)
    item_feature_list.append((new_song['id'], song_features))

    item_features_matrix_new = dataset.build_item_features(item_feature_list)

    # Refit the model with the updated item features
    model.fit(
        interactions_matrix, 
        item_features=item_features_matrix_new, 
        user_features=user_features_matrix,
        epochs=30, 
        num_threads=2
    )

    # Add a new user
    new_user = "dave"
    new_location = [51.5074, -0.1278]  # Example location
    new_user_feature = round_coords(new_location[0], new_location[1], precision=2)

    # Add the new user to the dataset
    dataset.fit_partial(users=[new_user], user_features=[new_user_feature])
    user_feature_list_new = user_feature_list + [(new_user, [new_user_feature])]

    # Rebuild user feature matrix with all users
    user_features_matrix_new = dataset.build_user_features(user_feature_list_new)

    # Build interactions again
    new_interaction = {"user_id": "dave", "song_id": songs_df.iloc[0]['id']}
    interactions_df = pd.concat([interactions_df, pd.DataFrame([new_interaction])], ignore_index=True)
    (interactions_matrix_new, _) = dataset.build_interactions(interactions_df.to_records(index=False))

    # Refit the model with the updated user features
    model.fit(
        interactions_matrix_new, 
        user_features=user_features_matrix_new, 
        item_features=item_features_matrix_new,
        epochs=30, 
        num_threads=2
    )

    # Generate recommendations
    print("Trail for Dave:", trail_recommendation("dave", model, dataset, user_features_matrix_new, item_features_matrix_new, interactions_df, song_plays))

def trail_recommendation(user_id, model, dataset, user_features_matrix, item_features_matrix, interactions_df, song_plays, top_n=3):
    user_idx = dataset.mapping()[0][user_id]
    song_idx_to_id = {v: k for k, v in dataset.mapping()[2].items()}

    scores = model.predict(user_idx,
                           np.arange(len(song_idx_to_id)),
                           user_features=user_features_matrix,
                           item_features=item_features_matrix)

    # Filter out already played songs
    already_played = interactions_df[interactions_df['user_id'] == user_id]['song_id'].tolist()
    already_played_idx = [dataset.mapping()[2][sid] for sid in already_played if sid in dataset.mapping()[2]]
    scores[already_played_idx] = -np.inf  # Remove them by setting to -inf

    top_items = np.argsort(-scores)[:top_n]
    trail = []

    # Fetch recommendations
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
            "why_recommend": "not implemented, need clustering",
        })
    return trail

if __name__ == "__main__":
    main()
