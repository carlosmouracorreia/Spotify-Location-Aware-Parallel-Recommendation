# Spotify Location-Aware Recommendation Demo - "GeoTracks"

This repository demonstrates an Efficient Recommendation System using Factorization Machines, a Recommender Model with Song Metadata using Spotify publicly provided audio features, and Synthesized user location features.

LightFM is a Recommender Model based on matrix factorization with feature-aware embeddings, and it is trained using gradient descent-style updates. These models are easily parallelized and distributed, which makes them a good fit for larger recommendation workloads.

The app can:

- recommend songs from the same genre when the requested track is present in the dataset
- add a missing song from user-provided audio features
- update a user's location-aware features and generate personalized recommendations

This is a strategy to recommend songs for a geolocation-aware item recommender interface/platform for academic purposes.

## Features Used

The demo uses the following input features:

- Song metadata: `artist`, `name`, `genre`
- Audio features: `valence`, `tempo`, `danceability`
- Location-aware user feature: rounded region from `latitude` / `longitude`


## Files

- `dataset_and_recommender_model.py` - initializes the LightFM dataset, builds user/item features, and trains the model.
- `main_song_recommendation_demo.py` - demo script that prompts for song and location input, then generates recommendations.

## Setup

1. Create a Python environment (recommended). The use of Anaconda or Miniconda is recommended because LightFM often relies on compiled binary dependencies, and conda simplifies installation and CPU parallelization support across OS's:

```bash
conda create -n spotify-recs python=3.11
conda activate spotify-recs
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Place the Spotify dataset at `data/ClassicHit15k.csv`.

A sample was provided with this repository under the MIT License and it's original source can be found here.
https://www.kaggle.com/datasets/thebumpkin/10400-classic-hits-10-genres-1923-to-2023


Alternatively, point the dataset path to a local file by setting:

```bash
export DATA_FILE_PATH=/path/to/ClassicHit15k.csv
```

1. Run the demo. The script will prompt for song and location values, with the ABBA example as the default.

```bash
python song_recommendation_demo.py
```

## Run the demo

```bash
python song_recommendation_demo.py
```

The script currently uses the hardcoded default song:

- Artist: `ABBA`
- Track: `Dancing Queen`

If the song is not present in the dataset, the script will prompt for audio features and add the song directly to the dataset.

## Notes

- The dataset loader uses a repo-relative path by default: `data/ClassicHit15k.csv`
- The demo is designed for repository sharing and can be extended with real user interaction flows.
- If the requested song is missing from the dataset, the app will prompt for audio features and add the song directly.
