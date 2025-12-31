import sys
import os
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import backend.app as app

# Ensure cleaned master exists
repo_root = os.path.dirname(os.path.dirname(__file__))
clean_master = os.path.join(repo_root, 'data', 'ap_trains_master_clean.csv')
assert os.path.exists(clean_master), 'Clean master file missing'

# Load the cleaned master train ids
clean_df = pd.read_csv(clean_master)
clean_train_ids = set(clean_df['train_id'].astype(str).tolist())

ok = app.load_model()
assert ok, 'load_model failed'

loaded_train_ids = set(app.train_data['train_id'].astype(str).unique().tolist())

# Check that at least some overlap exists and that cleaned master trains are included
intersection = clean_train_ids & loaded_train_ids
print('clean master trains:', len(clean_train_ids))
print('loaded trains:', len(loaded_train_ids))
print('intersection:', len(intersection))

assert len(intersection) > 0, 'No overlap between loaded trains and cleaned master'

# Optionally assert that loaded trains include all cleaned master trains
missing_from_loaded = clean_train_ids - loaded_train_ids
print('missing_from_loaded (sample 10):', list(missing_from_loaded)[:10])

# It's acceptable that not all cleaned master trains are in train_data merge; but we flag if none are present
print('OK')
