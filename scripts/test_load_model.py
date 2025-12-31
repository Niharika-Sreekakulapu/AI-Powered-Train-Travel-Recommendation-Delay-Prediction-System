import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import backend.app as app
ok = app.load_model()
print('load_model returned', ok)
if ok:
    print('train_data rows:', len(app.train_data))
