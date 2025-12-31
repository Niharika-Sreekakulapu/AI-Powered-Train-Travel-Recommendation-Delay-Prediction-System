import backend.app as app_module

print('Calling load_model()...')
app_module.load_model()
print('shap_available:', getattr(app_module, 'shap_available', None))
print('shap_explainer is None?', getattr(app_module, 'shap_explainer', None) is None)
print('shap_explainer_v2 is None?', getattr(app_module, 'shap_explainer_v2', None) is None)
print('model is None?', getattr(app_module, 'model', None) is None)
print('model_v2 is None?', getattr(app_module, 'model_v2', None) is None)
print('route_encoder is None?', getattr(app_module, 'route_encoder', None) is None)
print('route_encoder_v2 is None?', getattr(app_module, 'route_encoder_v2', None) is None)