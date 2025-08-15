import numpy as np
import torch

# --- Path to a MotionX .npy file ---
npy_file = "/home/lala/Documents/Data/Motion-Xplusplus/motion/motion_generation/smplx322/animation/comp_robot/zhangyuhong1/data/Motion-X++/v7/motion/motion_generation/smplx_322/animation/Ways_to_Catch_Factory_Line_clip2.npy"  # <-- replace

# --- Load and convert to tensor ---
motion_np = np.load(npy_file)
motion = torch.tensor(motion_np, dtype=torch.float32)

print(f"Loaded motion shape: {motion.shape}")

# --- Basic sanity check for expected SMPL-X columns ---
if motion.shape[1] < 312:
    raise ValueError(f"{npy_file}: expected >=312 columns, got {motion.shape[1]}")
else:
    print("Column count looks correct for SMPL-X body+betas.")

# --- Slice SMPL-X parameters (body-focused) ---
root_orient = motion[:, 0:3]
pose_body   = motion[:, 3:66]  # 63 body pose
pose_hand   = motion[:, 66:156]
pose_jaw    = motion[:, 156:159]
face_expr   = motion[:, 159:209]
face_shape  = motion[:, 209:309]
trans       = motion[:, 309:312]
betas       = motion[:, 312:]

print(f"root_orient shape: {root_orient.shape}")
print(f"pose_body shape:   {pose_body.shape}")
print(f"pose_hand shape:   {pose_hand.shape}")
print(f"pose_jaw shape:    {pose_jaw.shape}")
print(f"face_expr shape:   {face_expr.shape}")
print(f"face_shape shape:  {face_shape.shape}")
print(f"trans shape:       {trans.shape}")
print(f"betas shape:       {betas.shape}")
