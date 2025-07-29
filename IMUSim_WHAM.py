import torch
import numpy as np
import joblib
import os
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from imu_base_MOD import calculate_patch_IMU_signals


def process_and_save_patch_imu(pkl_file, imu_list, dt, model_root='smplpytorch/native/models'):
    """
    Processes SMPL data from a PKL file and saves synthetic IMU signals for specified body parts.

    Args:
        pkl_file (str): Path to the SMPL `.pkl` file.
        imu_list (dict): Dictionary specifying body parts and corresponding vertex indices.
        dt (float): Time step between frames.
        model_root (str): Path to SMPL model directory.
    """
    # Load SMPL data
    smpl_data = joblib.load(pkl_file)
    loaded_data = smpl_data[0]

    poses = loaded_data['pose']   # (N, 72)
    betas = loaded_data['betas']  # (N, 10)
    trans = loaded_data['trans']  # (N, 3)

    num_frames = poses.shape[0]

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root=model_root
    )

    cuda = torch.cuda.is_available()
    if cuda:
        smpl_layer.cuda()

    all_verts = []
    all_jtr = []

    for i in range(num_frames):
        pose_params = torch.tensor(poses[i:i+1], dtype=torch.float32)
        shape_params = torch.tensor(betas[0:1], dtype=torch.float32)

        if cuda:
            pose_params = pose_params.cuda()
            shape_params = shape_params.cuda()

        with torch.no_grad():
            verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)

        verts[:, :, 1] *= -1
        Jtr[:, :, 1] *= -1

        trans_torch = torch.tensor(trans[i:i+1], dtype=torch.float32)
        if cuda:
            trans_torch = trans_torch.cuda()

        verts = verts + trans_torch
        jtr = Jtr + trans_torch

        all_verts.append(verts.cpu().numpy())
        all_jtr.append(jtr.cpu().numpy())

    all_verts = np.array(all_verts)
    all_jtr = np.array(all_jtr)

    # Height scaling
    lists = {
        'forehead': [0, 1, 5, 132, 133, 232, 234, 235, 259, 335, 336, 3512, 3513, 3514, 3515, 3517, 3644, 3645, 3646, 3676, 3744, 3745, 3746, 3771],
        'right_leg': [847, 848, 849, 850, 872, 873, 874, 875, 876, 877, 904, 905, 906, 907, 957, 1159, 1365, 1366, 1499, 1500],
        'left_leg': [4333, 4334, 4335, 4336, 4358, 4359, 4360, 4361, 4362, 4363, 4645, 4648, 4711, 4712, 4801, 4802, 4839]
    }

    forehead_centroids = np.mean(all_verts[:, 0, lists['forehead']], axis=1)
    right_leg_centroids = np.mean(all_verts[:, 0, lists['right_leg']], axis=1)
    left_leg_centroids = np.mean(all_verts[:, 0, lists['left_leg']], axis=1)

    leg_centroids = (right_leg_centroids + left_leg_centroids) / 2
    distances = np.linalg.norm(forehead_centroids - leg_centroids, axis=1)
    average_height = np.mean(distances)
    scale_factor = 1.75 / average_height

    all_verts *= scale_factor
    all_jtr *= scale_factor

    # Save directory and base name
    original_dir = os.path.dirname(pkl_file)
    original_base = os.path.splitext(os.path.basename(pkl_file))[0]

    for body_part, config in imu_list.items():
        verts = config['verts']

        print(f"Processing PATCH IMU: {body_part}")

        positions, orientations, linear_velocity_world, linear_accel_loc, linear_accel_world, angular_velocity_world, angular_velocity_local, angular_accel_world = calculate_patch_IMU_signals(
            all_verts,
            verts,
            dt
        )

        save_filename = f"{original_base}_{body_part}_sIMU.npz"
        file_path = os.path.join(original_dir, save_filename)

        np.savez(
            file_path,
            positions=positions,
            orientations=orientations,
            global_acceleration=linear_accel_world,
            local_accel=linear_accel_loc,
            linear_velocity=linear_velocity_world,
            angular_velocity_world=angular_velocity_world,
            angular_velocity=angular_velocity_local,
            angular_acceleration=angular_accel_world
        )

        print(f"Saved PATCH IMU data: {file_path}")


UTD_lists = {
    'right_wrist': {'verts': [5669, 5705, 5430]},
    'right_thigh': {'verts': [847, 849, 957]}
}

pkl_path = '/media/lala/A/datasets/VIDIMU/smpl/VIDIMU/prepared/smpl/S01_A01_T01/wham_output.pkl'
FPS = 15

process_and_save_patch_imu(pkl_file = pkl_path, imu_list = UTD_lists, dt = 1/FPS)
