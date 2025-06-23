import numpy as np
from scipy.spatial.transform import Rotation as R

def build_patch_frame(v1, v2, v3):
    X = v2 - v1
    X /= np.linalg.norm(X)

    Z = np.cross(v2 - v1, v3 - v1)
    Z /= np.linalg.norm(Z)

    Y = np.cross(Z, X)

    # columns = local axes expressed in world coordinates (body → world)
    return np.stack([X, Y, Z], axis=1)          # (3, 3)


def calculate_patch_IMU_signals(
        all_verts,
        selected_vertices,
        dt,
        g_world=np.array([0.0, 0.0, -9.81])
):
    """
    Returns:
        centroids      : (N-1, 3)   – position of the patch (world)
        patch_quat     : (N-1, 4)   – world → patch quaternion  [x,y,z,w]
        a_local        : (N-1, 3)   – linear accel incl. g (patch frame)
        a_world        : (N-1, 3)   – linear accel incl. g (world frame)
        omega_world    : (N-1, 3)   – angular velocity  (world frame) [rad/s]
    """
    num_frames = all_verts.shape[0]

    # ------------------------------------------------------------------
    # 1) Build patch frame for every animation frame
    # ------------------------------------------------------------------
    centroids  = np.empty((num_frames, 3))
    frames_R   = np.empty((num_frames, 3, 3))     # body → world

    for f in range(num_frames):
        v1 = all_verts[f, 0, selected_vertices[0]]
        v2 = all_verts[f, 0, selected_vertices[1]]
        v3 = all_verts[f, 0, selected_vertices[2]]

        centroids[f] = (v1 + v2 + v3) / 3.0
        frames_R[f]  = build_patch_frame(v1, v2, v3)

    # ------------------------------------------------------------------
    # 2) Linear acceleration (world then local)
    # ------------------------------------------------------------------
    vel_world   = np.gradient(centroids, dt, axis=0)
    a_world_full = np.gradient(vel_world, dt, axis=0) + g_world          # (N, 3)

    # rotate to local frame
    a_local_full = (frames_R.transpose(0, 2, 1) @ a_world_full[..., None]).squeeze(-1)  # (N, 3)

    # ------------------------------------------------------------------
    # 3) Angular velocity (world coordinates)
    # ------------------------------------------------------------------
    R_rel = frames_R[1:] @ frames_R[:-1].transpose(0, 2, 1)              # (N-1, 3, 3)
    omega_world = R.from_matrix(R_rel).as_rotvec() / dt                  # (N-1, 3)

    # ------------------------------------------------------------------
    # 4) Orientation quaternion  **world → patch**
    #     (transpose turns body→world into world→body)
    # ------------------------------------------------------------------
    patch_quat_full = R.from_matrix(frames_R.transpose(0, 2, 1)).as_quat()   # (N, 4)

    # ------------------------------------------------------------------
    # 5) Trim first frame so everything lines up with omega_world
    # ------------------------------------------------------------------
    centroids_out  = centroids[1:]
    patch_quat_out = patch_quat_full[1:]
    a_local_out    = a_local_full[1:]
    a_world_out    = a_world_full[1:]

    return (
        centroids_out,      # position (world)
        patch_quat_out,     # orientation (world→patch)  [x,y,z,w]
        a_local_out,        # linear accel (patch frame)
        a_world_out,        # linear accel (world frame)
        omega_world         # angular velocity (world)
    )
