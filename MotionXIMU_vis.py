import os
import numpy as np
import torch
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from scipy.interpolate import interp1d

# === CONFIG ===
cuda = torch.cuda.is_available()
smpl_layer = SMPL_Layer(center_idx=0, gender='neutral', model_root='smplpytorch/native/models')
if cuda:
    smpl_layer = smpl_layer.cuda()

# === LOAD MOTION DATA ===
motion = np.load(r"C:\Users\lalas\Downloads\animation\comp_robot\zhangyuhong1\data\Motion-X++\v7\motion\motion_generation\smplx_322\animation\Ways_to_Jump_+_Sit_+_Fall_Reading_to_Children_clip1.npy")
motion = torch.tensor(motion).float()

motion_parms = {
    'root_orient': motion[:, :3],
    'pose_body': motion[:, 3:66],
    'pose_hand': motion[:, 66:156],
    'pose_jaw': motion[:, 156:159],
    'face_expr': motion[:, 159:209],
    'face_shape': motion[:, 209:309],
    'trans': motion[:, 309:312],
    'betas': motion[:, 312:]
}

# === SMPL COMPATIBILITY ===
zeros = torch.zeros(motion_parms['pose_body'].size(0), 6)
motion_parms['pose_body'] = torch.cat((motion_parms['pose_body'], zeros), dim=1)

# === FORWARD PASS THROUGH SMPL ===
all_verts = []
for i in range(motion_parms['root_orient'].shape[0]):
    global_orient = motion_parms['root_orient'][i]
    body_pose = motion_parms['pose_body'][i]
    body_shape = motion_parms['betas'][i]
    translation = motion_parms['trans'][i]

    pose_params = torch.cat([global_orient.view(1, -1), body_pose.view(1, -1)], dim=1)
    shape_params = body_shape.view(1, -1)

    if cuda:
        pose_params = pose_params.cuda()
        shape_params = shape_params.cuda()
        translation = translation.cuda()

    verts, _ = smpl_layer(pose_params, th_betas=shape_params)
    verts = 2 * verts + translation.view(1, 1, -1)
    all_verts.append(verts.cpu().detach().numpy())

all_verts = np.array(all_verts)[:, 0, :, :]  # shape: [frames, vertices, 3]

# === IMU VERTEX GROUPS ===
lists = {
    'right_wrist': [5405, 5430, 5431],
    'left_wrist': [1919, 1920, 1944],
    'right_shin': [4557, 4560, 4575],
    'left_shin': [1071, 1074, 1089],
    'right_thigh': [847, 849, 957],
    'left_thigh': [4333, 4335, 4645],
    'right_arm': [1546, 1547, 1569],
    'left_arm': [5015, 5017, 5039],
    'back': [726, 731, 1216],
    'chest': [1200, 1257, 1329]
}

# === IMU CALCULATION ===
def calculate_linear_acceleration_and_angular_velocity(all_verts, selected_vertices):
    num_frames = all_verts.shape[0]
    centroids = np.zeros((num_frames, 3))
    normals = np.zeros((num_frames, 3))

    for frame in range(num_frames):
        v1 = all_verts[frame, selected_vertices[0]]
        v2 = all_verts[frame, selected_vertices[1]]
        v3 = all_verts[frame, selected_vertices[2]]
        centroids[frame] = (v1 + v2 + v3) / 3.0
        normal = np.cross(v2 - v1, v3 - v1)
        normals[frame] = normal / np.linalg.norm(normal)

    velocities = np.diff(centroids, axis=0)
    linear_acceleration = np.diff(velocities, axis=0)
    normal_diff = np.diff(normals, axis=0)
    angular_velocity = np.cross(normal_diff, normals[1:])  # normals[1:] matches diff length

    return centroids, normals, linear_acceleration, angular_velocity

# === INTERPOLATION FUNCTION ===
def upsample(data, kind='cubic', factor=8):
    x_old = np.arange(data.shape[0])
    x_new = np.linspace(0, data.shape[0] - 1, data.shape[0] * factor)
    interp_fn = interp1d(x_old, data, axis=0, kind=kind)
    return interp_fn(x_new)

# === SAVE RESULTS ===
save_dir = r"C:\Users\lalas\Desktop\2"
os.makedirs(save_dir, exist_ok=True)

interp_factor = 8
window_size = 30

for body_part, verts in lists.items():
    positions, orientations, linear_accel, angular_vel = calculate_linear_acceleration_and_angular_velocity(all_verts, verts)

    # === Interpolate IMU data ===
    positions = upsample(positions, kind='cubic', factor=interp_factor)
    orientations = upsample(orientations, kind='linear', factor=interp_factor)
    linear_accel = upsample(linear_accel, kind='cubic', factor=interp_factor)
    angular_vel = upsample(angular_vel, kind='linear', factor=interp_factor)

    # === Save IMU data ===
    file_path = os.path.join(save_dir, f'{body_part}_1.npz')
    np.savez(file_path,
             positions=positions,
             orientations=orientations,
             linear_acceleration=linear_accel,
             angular_velocity=angular_vel)
    print(f"Saved IMU data: {file_path}")

    # === Create Linear Acceleration Sliding Window Video ===
    num_frames = linear_accel.shape[0] - window_size
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f'Linear Acceleration - {body_part}')
    ax.set_xlim(0, window_size - 1)
    ax.set_ylim(np.min(linear_accel) - 0.5, np.max(linear_accel) + 0.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Acceleration')
    line_x, = ax.plot([], [], label='X')
    line_y, = ax.plot([], [], label='Y')
    line_z, = ax.plot([], [], label='Z')
    ax.legend()

    def update_accel(frame):
        x = np.arange(window_size)
        line_x.set_data(x, linear_accel[frame:frame + window_size, 0])
        line_y.set_data(x, linear_accel[frame:frame + window_size, 1])
        line_z.set_data(x, linear_accel[frame:frame + window_size, 2])
        return line_x, line_y, line_z

    ani_acc = FuncAnimation(fig, update_accel, frames=num_frames, blit=True)
    acc_video_path = os.path.join(save_dir, f'{body_part}_linear_acceleration.mp4')
    ani_acc.save(acc_video_path, writer=FFMpegWriter(fps=30))
    plt.close()

    # === Create Angular Velocity Sliding Window Video ===
    num_frames = angular_vel.shape[0] - window_size
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.set_title(f'Angular Velocity - {body_part}')
    ax.set_xlim(0, window_size - 1)
    ax.set_ylim(np.min(angular_vel) - 0.5, np.max(angular_vel) + 0.5)
    ax.set_xlabel('Frame')
    ax.set_ylabel('Angular Velocity')
    line_x, = ax.plot([], [], label='X')
    line_y, = ax.plot([], [], label='Y')
    line_z, = ax.plot([], [], label='Z')
    ax.legend()

    def update_ang(frame):
        x = np.arange(window_size)
        line_x.set_data(x, angular_vel[frame:frame + window_size, 0])
        line_y.set_data(x, angular_vel[frame:frame + window_size, 1])
        line_z.set_data(x, angular_vel[frame:frame + window_size, 2])
        return line_x, line_y, line_z

    ani_ang = FuncAnimation(fig, update_ang, frames=num_frames, blit=True)
    ang_video_path = os.path.join(save_dir, f'{body_part}_angular_velocity.mp4')
    ani_ang.save(ang_video_path, writer=FFMpegWriter(fps=30))
    plt.close()

    print(f"Saved videos for {body_part}")

print("Processing complete.")
