import os
import pandas as pd

from dataset_and_recommender_model import (
    songs_df,
    model,
    dataset,
    user_features_matrix,
    item_features_matrix,
    user_feature_list,
    interactions_df,
    song_plays,
    round_coords,
    trail_recommendation,
)

DEFAULT_ARTIST = 'ABBA'
DEFAULT_TRACK = 'Dancing Queen'
DEFAULT_GENRE = 'pop'
DEFAULT_VALENCE = 0.8
DEFAULT_TEMPO = 120
DEFAULT_DANCEABILITY = 0.8
DEFAULT_ENERGY = 0.7
DEFAULT_USER = 'dave'
DEFAULT_USER_LOCATION = {'lat': 40.7128, 'lon': -74.0060, 'city': 'NYC'}


def find_song_in_dataset(artist_name, song_name):
    match = songs_df[(songs_df['name'] == song_name) & (songs_df['artist'] == artist_name)]
    return match.iloc[0] if not match.empty else None


def recommend_most_similar_on_same_genre(genre, top_n=5):
    if genre is None:
        return pd.DataFrame(columns=songs_df.columns)

    exact_match = songs_df[songs_df['genre'].str.lower() == genre.lower()]
    if not exact_match.empty:
        return exact_match.sample(n=min(top_n, len(exact_match)))

    contains_match = songs_df[songs_df['genre'].str.contains(genre, case=False, na=False)]
    if not contains_match.empty:
        return contains_match.sample(n=min(top_n, len(contains_match)))

    return pd.DataFrame(columns=songs_df.columns)


def prompt_with_default(prompt_text, default_value):
    answer = input(f"{prompt_text} [{default_value}]: ").strip()
    return answer if answer else default_value


def prompt_float(prompt_text, default_value):
    while True:
        answer = input(f"{prompt_text} [{default_value}]: ").strip()
        if not answer:
            return default_value
        try:
            return float(answer)
        except ValueError:
            print('Please enter a valid number.')


def build_song_features(song):
    return [
        f"valence:{round(song['valence'], 2)}",
        f"tempo:{int(song['tempo'])}",
        f"danceability:{round(song['danceability'], 2)}",
    ]


def ensure_song_in_dataset(song_info, item_feature_list):
    existing = find_song_in_dataset(song_info['artist'], song_info['name'])
    if existing is not None:
        return existing, item_features_matrix

    song_id = f"song{len(songs_df) + 1}"
    song_row = {
        'id': song_id,
        'name': song_info['name'],
        'artist': song_info['artist'],
        'danceability': song_info['danceability'],
        'energy': song_info['energy'],
        'valence': song_info['valence'],
        'tempo': song_info['tempo'],
        'genre': song_info['genre'],
    }
    songs_df.loc[len(songs_df)] = song_row

    item_features = build_song_features(song_info)
    dataset.fit_partial(items=[song_id], item_features=item_features)
    item_feature_list.append((song_id, item_features))
    updated_item_features = dataset.build_item_features(item_feature_list)

    return songs_df[songs_df['id'] == song_id].iloc[0], updated_item_features


def input_song_info():
    artist = prompt_with_default('Artist', DEFAULT_ARTIST)
    track = prompt_with_default('Track', DEFAULT_TRACK)
    song = find_song_in_dataset(artist, track)
    if song is not None:
        print(f"Found song in dataset: {song['name']} by {song['artist']}")
        return song, item_features_matrix

    print('Song not found in dataset. Please enter the song metadata and audio features.')
    genre = prompt_with_default('Genre', DEFAULT_GENRE)
    valence = prompt_float('Valence', DEFAULT_VALENCE)
    tempo = prompt_float('Tempo', DEFAULT_TEMPO)
    danceability = prompt_float('Danceability', DEFAULT_DANCEABILITY)
    energy = prompt_float('Energy', DEFAULT_ENERGY)

    song_info = {
        'artist': artist,
        'name': track,
        'genre': genre,
        'valence': valence,
        'tempo': tempo,
        'danceability': danceability,
        'energy': energy,
    }
    return ensure_song_in_dataset(song_info, item_feature_list)


def ensure_user_features(user_id, location, user_feature_list):
    region_feature = round_coords(location['lat'], location['lon'])
    if user_id in dataset.mapping()[0]:
        dataset.fit_partial(users=[user_id], user_features=[region_feature])
        for idx, (uid, features) in enumerate(user_feature_list):
            if uid == user_id:
                user_feature_list[idx] = (user_id, [region_feature])
                break
        else:
            user_feature_list.append((user_id, [region_feature]))
    else:
        dataset.fit_partial(users=[user_id], user_features=[region_feature])
        user_feature_list.append((user_id, [region_feature]))

    return dataset.build_user_features(user_feature_list)


def add_user_song_interaction(interactions_df, user_id, song_id):
    if not ((interactions_df['user_id'] == user_id) & (interactions_df['song_id'] == song_id)).any():
        interaction = {'user_id': user_id, 'song_id': song_id}
        return pd.concat([interactions_df, pd.DataFrame([interaction])], ignore_index=True)
    return interactions_df


def print_recommendations(recommendations, user_id):
    if not recommendations:
        print(f'No recommendations available for {user_id}.')
        return

    print(f'Recommended songs for {user_id}:')
    for rec in recommendations:
        print(
            f"- {rec['song_name']} by {rec['artist']} (Last played by: {rec['last_user_playing']}, "
            f"City: {rec['city']}, Region: {rec['region']})"
        )


def main():
    print('=== Spotify Location-Aware Recommendation Demo ===')
    print('Enter a song and location to get recommendations. Leave blank to use the default ABBA example.')

    seed_song, active_item_features = input_song_info()

    user_location = {
        'lat': prompt_float('User latitude', DEFAULT_USER_LOCATION['lat']),
        'lon': prompt_float('User longitude', DEFAULT_USER_LOCATION['lon']),
        'city': prompt_with_default('User city', DEFAULT_USER_LOCATION['city']),
    }

    print(f"Seed song genre: {seed_song['genre']}")
    print(f"Seed song features: valence={seed_song['valence']}, tempo={seed_song['tempo']}, danceability={seed_song['danceability']}")

    song_id = seed_song['id']
    if song_id not in dataset.mapping()[2]:
        print(f"Song '{song_id}' is not present in the recommendation dataset.")
        return

    updated_interactions = add_user_song_interaction(interactions_df, DEFAULT_USER, song_id)
    user_features = ensure_user_features(DEFAULT_USER, user_location, user_feature_list)

    song_plays.append({
        'song': song_id,
        'user': DEFAULT_USER,
        'city': user_location['city'],
        'region': round_coords(user_location['lat'], user_location['lon']),
        'timestamp': pd.Timestamp.now(),
    })

    interactions_matrix_new, _ = dataset.build_interactions(updated_interactions.to_records(index=False))
    model.fit(
        interactions_matrix_new,
        user_features=user_features,
        item_features=active_item_features,
        epochs=30,
        num_threads=2,
    )

    print(f'Generating recommendations for {DEFAULT_USER}...')
    recommendations = trail_recommendation(
        DEFAULT_USER,
        model,
        dataset,
        user_features,
        active_item_features,
        updated_interactions,
        song_plays,
        top_n=5,
    )

    print_recommendations(recommendations, DEFAULT_USER)


if __name__ == '__main__':
    main()
