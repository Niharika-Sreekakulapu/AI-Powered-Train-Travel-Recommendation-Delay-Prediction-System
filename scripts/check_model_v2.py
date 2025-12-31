import joblib, os
m='backend/model_v2.pkl'
print('exists', os.path.exists(m))
try:
    mod=joblib.load(m)
    print('loaded type', type(mod))
    if hasattr(mod,'feature_names_in_'):
        print('feature_names count', len(getattr(mod,'feature_names_in_',[])))
        print('sample features:', getattr(mod,'feature_names_in_')[:10])
    else:
        print('no feature_names_in_')
except Exception as e:
    print('load error', e)
