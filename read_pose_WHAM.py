import numpy as np
import joblib

# Load data
data = joblib.load('Lars/demo/wham_output.pkl')

# Convert NumPy int64 keys to regular integers (if necessary)
data = {int(k): v for k, v in data.items()}

# List of keys to stack
keys_to_stack = ["pose", "trans", "pose_world", "trans_world", "betas", "verts", "frame_ids"]

# Stack each key across all outer keys (0,1,2)
stacked_data = {key: np.concatenate([data[i][key] for i in sorted(data.keys())], axis=0) for key in keys_to_stack}

# Save as compressed NPZ file
np.savez_compressed("Lars.npz", **stacked_data)

print("Saved stacked data as 'stacked_data.npz'")
