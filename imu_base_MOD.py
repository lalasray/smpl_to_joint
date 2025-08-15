import numpy as np
from scipy.spatial.transform import Rotation as R
from scipy.signal import savgol_filter
from scipy.linalg import logm

def build_patch_frame(v1, v2, v3, fallback=None):
    X = v2 - v1
    X /= np.linalg.norm(X)

    Z = np.cross(v2 - v1, v3 - v1)
    Z_norm = np.linalg.norm(Z)
    if Z_norm < 1e-8:
        if fallback is not None:
            return fallback
        else:
            raise ValueError("Degenerate triangle: vertices are collinear or nearly so.")
    Z /= Z_norm
    Y = np.cross(Z, X)
    R_frame = np.stack([X, Y, Z], axis=1)
    return R_frame

def robust_derivative(data, dt, framerate=None, auto_jitter=True):
    if framerate is None:
        framerate = 1.0 / dt
    
    # Auto-estimate jitter: heuristic based on signal standard deviation
    signal_std = np.std(data, axis=0).mean()
    delta_std = np.std(np.diff(data, axis=0), axis=0).mean()
    
    if auto_jitter:
        jitter_factor = min(max(delta_std / (signal_std + 1e-8), 0.5), 2.0)  # clamp
    else:
        jitter_factor = 1.0

    approx_window = int(framerate * 0.1 * jitter_factor)
    window_length = max(5, approx_window | 1)  # odd, at least 5
    polyorder = 2 if window_length > 5 else 1

    return savgol_filter(data, window_length, polyorder, deriv=1, delta=dt, axis=0, mode='interp')

def calculate_patch_IMU_signals_MotionX(
    all_verts, selected_vertices, dt,
    g_world=np.array([0, 0, -9.81]),
    framerate=None,
    auto_jitter=True
):
    num_frames = all_verts.shape[0]

    centroids = np.zeros((num_frames, 3))
    frames_R = []
    last_valid_R = None

    for frame in range(num_frames):
        v1 = all_verts[frame,selected_vertices[0], :]
        v2 = all_verts[frame,selected_vertices[1], :]
        v3 = all_verts[frame,selected_vertices[2], :]

        centroids[frame] = (v1 + v2 + v3) / 3.0

        try:
            R_frame = build_patch_frame(v1, v2, v3, fallback=last_valid_R)
        except Exception:
            R_frame = last_valid_R

        frames_R.append(R_frame)
        last_valid_R = R_frame

    frames_R = np.stack(frames_R)

    # Linear velocities and accelerations in world frame
    velocities = robust_derivative(centroids, dt, framerate, auto_jitter)
    linear_accel_world = robust_derivative(velocities, dt, framerate, auto_jitter)

    a_total_world = linear_accel_world + g_world
    linear_accel_local = np.einsum('nij,nj->ni', frames_R.transpose(0, 2, 1), a_total_world)

    # Angular velocity: both in world and local
    angular_velocity_world = []
    angular_velocity_local = []

    for i in range(num_frames - 1):
        R1 = frames_R[i]
        R2 = frames_R[i + 1]

        R_rel = R2 @ R1.T
        log_R = logm(R_rel).real
        omega_hat = log_R / dt

        omega_global = np.array([
            omega_hat[2, 1],
            omega_hat[0, 2],
            omega_hat[1, 0]
        ])

        omega_local = R1.T @ omega_global

        angular_velocity_world.append(omega_global)
        angular_velocity_local.append(omega_local)

    angular_velocity_world = np.vstack(angular_velocity_world)
    angular_velocity_local = np.vstack(angular_velocity_local)

    # Angular acceleration in world frame
    angular_accel_world = robust_derivative(angular_velocity_world, dt, framerate, auto_jitter)

    # Quaternions (absolute)
    patch_quat_absolute = R.from_matrix(frames_R).as_quat()
    patch_quat_absolute = patch_quat_absolute[:-1]  # match other signals

    return centroids[:-1], patch_quat_absolute, velocities[:-1], linear_accel_local[:-1], linear_accel_world[:-1], angular_velocity_world, angular_velocity_local, angular_accel_world

def calculate_patch_IMU_signals(
    all_verts, selected_vertices, dt,
    g_world=np.array([0, 0, -9.81]),
    framerate=None,
    auto_jitter=True
):
    num_frames = all_verts.shape[0]

    centroids = np.zeros((num_frames, 3))
    frames_R = []
    last_valid_R = None

    for frame in range(num_frames):
        v1 = all_verts[frame, 0, selected_vertices[0], :]
        v2 = all_verts[frame, 0, selected_vertices[1], :]
        v3 = all_verts[frame, 0, selected_vertices[2], :]

        centroids[frame] = (v1 + v2 + v3) / 3.0

        try:
            R_frame = build_patch_frame(v1, v2, v3, fallback=last_valid_R)
        except Exception:
            R_frame = last_valid_R

        frames_R.append(R_frame)
        last_valid_R = R_frame

    frames_R = np.stack(frames_R)

    # Linear velocities and accelerations in world frame
    velocities = robust_derivative(centroids, dt, framerate, auto_jitter)
    linear_accel_world = robust_derivative(velocities, dt, framerate, auto_jitter)

    a_total_world = linear_accel_world + g_world
    linear_accel_local = np.einsum('nij,nj->ni', frames_R.transpose(0, 2, 1), a_total_world)

    # Angular velocity: both in world and local
    angular_velocity_world = []
    angular_velocity_local = []

    for i in range(num_frames - 1):
        R1 = frames_R[i]
        R2 = frames_R[i + 1]

        R_rel = R2 @ R1.T
        log_R = logm(R_rel).real
        omega_hat = log_R / dt

        omega_global = np.array([
            omega_hat[2, 1],
            omega_hat[0, 2],
            omega_hat[1, 0]
        ])

        omega_local = R1.T @ omega_global

        angular_velocity_world.append(omega_global)
        angular_velocity_local.append(omega_local)

    angular_velocity_world = np.vstack(angular_velocity_world)
    angular_velocity_local = np.vstack(angular_velocity_local)

    # Angular acceleration in world frame
    angular_accel_world = robust_derivative(angular_velocity_world, dt, framerate, auto_jitter)

    # Quaternions (absolute)
    patch_quat_absolute = R.from_matrix(frames_R).as_quat()
    patch_quat_absolute = patch_quat_absolute[:-1]  # match other signals

    return centroids[:-1], patch_quat_absolute, velocities[:-1], linear_accel_local[:-1], linear_accel_world[:-1], angular_velocity_world, angular_velocity_local, angular_accel_world

