import torch
import numpy as np
import joblib
import os
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from imu_base_MOD import calculate_patch_IMU_signals_MotionX

def process_and_save_patch_imu_motionx(
    npy_file, imu_list, dt, model_root='smplpytorch/native/models'
):
    """
    Processes MotionX SMPL-X `.npy` data and saves synthetic IMU signals
    for specified body parts (WHAM-style outputs).

    Args:
        npy_file (str): Path to MotionX `.npy` file containing SMPL-X params.
        imu_list (dict): {'part': {'verts': [v_i, ...]}, ...}
        dt (float): Time step between frames.
        model_root (str): Path to SMPL model directory.
    """
    # --- Load MotionX data (N x D) ---
    motion_np = np.load(npy_file)
    motion = torch.tensor(motion_np, dtype=torch.float32)

    # Sanity check for expected columns
    if motion.shape[1] < 312:
        raise ValueError(f"{npy_file}: expected >=312 columns, got {motion.shape[1]}")

    # Slice SMPL-X parameters from MotionX layout
    root_orient = motion[:, 0:3]          # (N, 3)
    pose_body   = motion[:, 3:66]         # (N, 63)
    betas       = motion[:, 312:]         # (N, ?)
    trans       = motion[:, 309:312]      # (N, 3)

    # Build SMPL pose (72 = 3 root + 69 body). Add 6 zeros for missing joints
    zeros6 = torch.zeros(pose_body.size(0), 6, dtype=motion.dtype, device=motion.device)
    pose_body_69 = torch.cat([pose_body, zeros6], dim=1)
    pose_params = torch.cat([root_orient, pose_body_69], dim=1)  # (N, 72)

    # --- SMPL layer ---
    smpl_layer = SMPL_Layer(center_idx=0, gender='neutral', model_root=model_root)
    if torch.cuda.is_available():
        smpl_layer = smpl_layer.cuda()
        pose_params = pose_params.cuda()
        betas = betas.cuda()
        trans = trans.cuda()

    with torch.no_grad():
        verts, Jtr = smpl_layer(pose_params, th_betas=betas)  # (N, V, 3), (N, 24, 3)

    # Apply translation
    verts = verts + trans[:, None, :]
    Jtr   = Jtr   + trans[:, None, :]

    # Convert to numpy
    all_verts = verts.detach().cpu().numpy()
    all_jtr   = Jtr.detach().cpu().numpy()

    # --- Height scaling (normalize to 1.75 m) ---
    ref_lists = {
        'forehead': [0, 1, 5, 132, 133, 232, 234, 235, 259, 335, 336,
                     3512, 3513, 3514, 3515, 3517, 3644, 3645, 3646,
                     3676, 3744, 3745, 3746, 3771],
        'right_leg': [847, 848, 849, 850, 872, 873, 874, 875, 876, 877,
                      904, 905, 906, 907, 957, 1159, 1365, 1366, 1499, 1500],
        'left_leg': [4333, 4334, 4335, 4336, 4358, 4359, 4360, 4361, 4362, 4363,
                     4645, 4648, 4711, 4712, 4801, 4802, 4839],
    }

    forehead_centroids = np.mean(all_verts[:, ref_lists['forehead'], :], axis=1)
    right_leg_centroids = np.mean(all_verts[:, ref_lists['right_leg'], :], axis=1)
    left_leg_centroids  = np.mean(all_verts[:, ref_lists['left_leg'], :], axis=1)

    leg_centroids = (right_leg_centroids + left_leg_centroids) / 2
    distances = np.linalg.norm(forehead_centroids - leg_centroids, axis=1)
    avg_height = np.mean(distances)

    if avg_height > 0:
        scale = 1.75 / avg_height
        all_verts *= scale
        all_jtr   *= scale

    # --- Save IMU outputs per body part ---
    out_dir  = os.path.dirname(npy_file)
    out_base = os.path.splitext(os.path.basename(npy_file))[0]

    for body_part, cfg in imu_list.items():
        print(f"Processing PATCH IMU: {body_part}")
        verts_idx = cfg['verts']

        (positions, orientations, lin_vel_world, lin_acc_loc, lin_acc_world,
         ang_vel_world, ang_vel_local, ang_acc_world) = calculate_patch_IMU_signals_MotionX(
            all_verts, verts_idx, dt
        )

        save_path = os.path.join(out_dir, f"{out_base}_{body_part}_sIMU.npz")
        np.savez(
            save_path,
            positions=positions,
            orientations=orientations,
            global_acceleration=lin_acc_world,
            local_accel=lin_acc_loc,
            linear_velocity=lin_vel_world,
            angular_velocity_world=ang_vel_world,
            angular_velocity=ang_vel_local,
            angular_acceleration=ang_acc_world,
        )
        print(f"Saved PATCH IMU data: {save_path}")


def process_all_npy_files_in_dir(root_dir, imu_list, fps=15):
    """
    Recursively process all MotionX .npy files in a directory with given IMU config and FPS.

    Args:
        root_dir (str): Root directory to search.
        imu_list (dict): IMU configuration for body parts.
        fps (int): Frames per second for motion data.
    """
    dt = 1 / fps

    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.npy'):
                npy_path = os.path.join(dirpath, filename)
                try:
                    process_and_save_patch_imu_motionx(
                        npy_file=npy_path,
                        imu_list=imu_list,
                        dt=dt
                    )
                except Exception as e:
                    print(f"[ERROR] Failed to process {npy_path}: {e}")



def main():
    root_dir = '/home/lala/Documents/Data/Motion-Xplusplus/motion/motion_generation/smplx322'


    Motionx_lists = {
        'right_wrist': {'verts': [5669, 5705, 5430]},
        'right_thigh': {'verts': [847, 849, 957]},
        'left_wrist': {'verts': [1961, 1969, 2244]},
        'left_thigh': {'verts': [4334,4333,4336]},
    }
    FPS = 30

    process_all_npy_files_in_dir(root_dir=root_dir, imu_list=Motionx_lists, fps=FPS)


if __name__ == "__main__":
    main()