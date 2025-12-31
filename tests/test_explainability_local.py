import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from backend.app import _generate_shap_explanation_for_features


def test_generate_shap_explanation_local():
    # Simple synthetic model and data
    X = np.array([
        [10, 1, 0],
        [200, 6, 1],
        [50, 3, 0]
    ])
    y = np.array([5.0, 120.0, 20.0])
    feature_names = ['distance_km', 'day_of_week', 'season_encoded']

    model = DecisionTreeRegressor(random_state=42)
    model.fit(X, y)

    # Tree SHAP explainer
    try:
        import shap
        explainer = shap.TreeExplainer(model)
    except Exception:
        # If shap not available, skip test
        return

    features_df = pd.DataFrame([X[1, :]], columns=feature_names)
    res = _generate_shap_explanation_for_features(model, explainer, features_df)

    assert isinstance(res, dict)
    assert 'top_features' in res
    assert 'feature_contributions' in res
    # contributions sum to ~100 (allow small numerical differences)
    total = sum(res['feature_contributions'].values()) if res['feature_contributions'] else 0
    assert abs(total - 100.0) < 1e-6 or total == 0.0