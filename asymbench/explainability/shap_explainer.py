import shap


def explain_model(model, X_train):
    explainer = shap.Explainer(model, X_train)
    shap_values = explainer(X_train[:100])

    return {"shap_values": shap_values.values.tolist()}
