import numpy as np
from scipy.interpolate import CubicHermiteSpline
from scipy.ndimage import gaussian_filter1d

def upsample_smpl_with_jitter(
    poses,
    trans,
    fps_original=30,
    fps_high=200,
    pose_jitter_std=0.001,  # radians
    trans_jitter_std=0.001,  # meters
    jitter_smooth_sigma=5,
    seed=42
):
    """
    Upsample SMPL pose (N, 72) and trans (N, 3) to higher frequency using
    Hermite spline with finite-difference velocity, then add smooth jitter.
    
    Returns:
      poses_high: (M, 72)
      trans_high: (M, 3)
      t_high: (M,)
    """
    rng = np.random.default_rng(seed)

    num_frames = poses.shape[0]
    duration = (num_frames - 1) / fps_original
    t_original = np.linspace(0, duration, num_frames)
    num_frames_high = int(duration * fps_high) + 1
    t_high = np.linspace(0, duration, num_frames_high)

    # --- 1) Hermite spline for translation ---
    trans_velocity = np.gradient(trans, t_original, axis=0)
    trans_high = np.zeros((num_frames_high, 3))
    for d in range(3):
        hermite = CubicHermiteSpline(t_original, trans[:, d], trans_velocity[:, d])
        trans_high[:, d] = hermite(t_high)

    # --- 2) Hermite spline for pose ---
    poses_velocity = np.gradient(poses, t_original, axis=0)
    poses_high = np.zeros((num_frames_high, poses.shape[1]))
    for d in range(poses.shape[1]):
        hermite = CubicHermiteSpline(t_original, poses[:, d], poses_velocity[:, d])
        poses_high[:, d] = hermite(t_high)

    # --- 3) Add smooth jitter ---
    pose_jitter = rng.normal(0, pose_jitter_std, poses_high.shape)
    pose_jitter = gaussian_filter1d(pose_jitter, sigma=jitter_smooth_sigma, axis=0)
    poses_high += pose_jitter

    trans_jitter = rng.normal(0, trans_jitter_std, trans_high.shape)
    trans_jitter = gaussian_filter1d(trans_jitter, sigma=jitter_smooth_sigma, axis=0)
    trans_high += trans_jitter

    return poses_high, trans_high, t_high

