import numpy as np
from scipy.spatial.transform import Rotation as R

def build_patch_frame(v1, v2, v3):
    """
    Build a local right-handed frame from 3 vertices:
      X axis: edge vector (v2 - v1)
      Z axis: normal of the triangle
      Y axis: orthogonal to X and Z
    """
    X = v2 - v1
    X /= np.linalg.norm(X)

    Z = np.cross(v2 - v1, v3 - v1)
    Z /= np.linalg.norm(Z)

    Y = np.cross(Z, X)

    R_frame = np.stack([X, Y, Z], axis=1)  # (3, 3)
    return R_frame


def calculate_patch_IMU_signals(all_verts, selected_vertices, dt, g_world=np.array([0, 0, -9.81])):
    """
    Compute realistic IMU signals for a patch on the mesh:
      - Position: patch centroid
      - Linear acceleration: includes gravity, rotated to local frame
      - Orientation: local patch frame as quaternion
      - Angular velocity: from patch frame rotation change

    all_verts: (N, 1, V, 3)
    selected_vertices: list of 3 vertex indices
    dt: timestep in seconds
    g_world: gravity vector in world frame (default: Z down)
    """
    num_frames = all_verts.shape[0]

    centroids = np.zeros((num_frames, 3))
    frames_R = []

    for frame in range(num_frames):
        v1 = all_verts[frame, 0, selected_vertices[0], :]
        v2 = all_verts[frame, 0, selected_vertices[1], :]
        v3 = all_verts[frame, 0, selected_vertices[2], :]

        centroids[frame] = (v1 + v2 + v3) / 3.0

        R_frame = build_patch_frame(v1, v2, v3)
        frames_R.append(R_frame)

    frames_R = np.stack(frames_R)  # (N, 3, 3)

    #Kinematic acceleration in world frame (motion only)
    velocities = np.gradient(centroids, dt, axis=0)
    linear_accel_world = np.gradient(velocities, dt, axis=0)

    #Add gravity in world frame
    a_total_world = linear_accel_world + g_world  # broadcasts automatically

    #Rotate total (motion + gravity) to patch local frame
    linear_accel_local = []
    for i in range(num_frames):
        R_patch = frames_R[i]
        a_world = a_total_world[i]
        a_local = R_patch.T @ a_world  # world -> local
        linear_accel_local.append(a_local)
    linear_accel_local = np.vstack(linear_accel_local)

    #Patch angular velocity: relative rotation matrix log
    angular_velocity = []
    for i in range(num_frames - 1):
        R1 = frames_R[i]
        R2 = frames_R[i + 1]

        R_rel = R2 @ R1.T
        rotvec = R.from_matrix(R_rel).as_rotvec()
        omega = rotvec / dt
        angular_velocity.append(omega)
    angular_velocity = np.vstack(angular_velocity)  # (N-1, 3)

    #Patch orientation as quaternion (for saving)
    patch_quat = R.from_matrix(frames_R).as_quat()
    patch_quat = patch_quat[:-1]  # match angular velocity length

    #Trim position and linear accel to match length
    centroids_out = centroids[:-1]
    linear_accel_local_out = linear_accel_local[:-1]

    return centroids_out, patch_quat, linear_accel_local_out, linear_accel_world, angular_velocity
