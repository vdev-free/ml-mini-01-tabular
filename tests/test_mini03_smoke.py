from mini03.data import make_customers
from mini03.features import scale_features
from mini03.cluster import fit_kmeans


def test_mini03_runs():
    df = make_customers(n=200)
    scaled_df, _ = scale_features(df)
    model, labels = fit_kmeans(scaled_df, k=3)

    assert len(labels) == 200
    assert model.n_clusters == 3
