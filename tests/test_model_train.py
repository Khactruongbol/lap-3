import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.clustering import ClusteringCandidate
from src.model_train import save_model_artifact
from src.preprocessing import FeatureSet


def test_save_model_artifact_contains_required_metadata(tmp_path):
    model_file = tmp_path / "model.joblib"
    profile_path = tmp_path / "profile.csv"
    metrics_path = tmp_path / "metrics.csv"
    clustered = pd.DataFrame({"customer_id": [1, 2], "cluster": [0, 1]})
    scaler = StandardScaler().fit([[0.0], [1.0]])
    feature_sets = {
        "demo": FeatureSet(
            name="demo",
            matrix=np.array([[0.0], [1.0]]),
            feature_names=["feature"],
            scaler=scaler,
        )
    }
    best = ClusteringCandidate(
        algorithm="kmeans",
        feature_set="demo",
        params={"n_clusters": 2},
        labels=np.array([0, 1]),
        inertia=0.0,
        noise_ratio=0.0,
    )

    save_model_artifact(model_file, "mall", best, feature_sets, clustered, profile_path, metrics_path)

    import joblib

    artifact = joblib.load(model_file)
    assert artifact["algorithm"] == "kmeans"
    assert artifact["feature_set"] == "demo"
    assert artifact["feature_names"] == ["feature"]
    assert artifact["labels"] == [0, 1]
    assert artifact["profile_path"] == profile_path.as_posix()
