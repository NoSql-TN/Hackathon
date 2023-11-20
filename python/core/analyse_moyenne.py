def get_moyenne_pop(file):
    return file['popularity'].mean()

def mean_pop_per_genre(file):
    return file.groupby('track_genre')['popularity'].mean().sort_values(ascending=False)

def duration_popularity(file):
    file['duration_s'] = file['duration_ms'] // 1000
    return file.groupby('duration_s')['popularity'].mean().sort_values(ascending=False)