import os
import numpy as np
import torch
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
import glob

# Initialize SMPL layer
cuda = torch.cuda.is_available()
smpl_layer = SMPL_Layer(center_idx=0, gender='neutral', model_root='smplpytorch/native/models')
if cuda:
    smpl_layer = smpl_layer.cuda()

# File paths
motion_file_path = r'/media/lala/Crucial X62/CrosSim/motionx_other'
npy_files = glob.glob(os.path.join(motion_file_path, '**', '*.npy'), recursive=True)
absolute_paths = [os.path.abspath(file) for file in npy_files]

# Define vertex groups
lists = {
    'forehead': [0, 1, 5, 132, 133, 232, 234, 235, 259, 335, 336, 3512, 3513, 3514, 3515, 3517, 3644, 3645, 3646, 3676, 3744, 3745, 3746, 3771],
    'right_leg': [847, 848, 849, 850, 872, 873, 874, 875, 876, 877, 904, 905, 906, 907, 957, 1159, 1365, 1366, 1499, 1500],
    'left_leg': [4333, 4334, 4335, 4336, 4358, 4359, 4360, 4361, 4362, 4363, 4645, 4648, 4711, 4712, 4801, 4802, 4839]
}

# Process motion files
for path in absolute_paths:
    try:
        motion = np.load(path)
        motion = torch.tensor(motion).float()

        motion_parms = {
            'root_orient': motion[:, :3],
            'pose_body': motion[:, 3:3 + 63],
            'pose_hand': motion[:, 66:66 + 90],
            'pose_jaw': motion[:, 66 + 90:66 + 93],
            'face_expr': motion[:, 159:159 + 50],
            'face_shape': motion[:, 209:209 + 100],
            'trans': motion[:, 309:309 + 3],
            'betas': motion[:, 312:],
        }

    except Exception as e:
        print(f"Failed to process file: {path}. Error: {str(e)}")
        continue

    zeros = torch.zeros(motion_parms['pose_body'].size(0), 6)
    motion_parms['pose_body'] = torch.cat((motion_parms['pose_body'], zeros), dim=1)

    all_verts = []
    all_jtr = []

    for frame_index in range(motion_parms['root_orient'].shape[0]):
        global_orient = motion_parms['root_orient'][frame_index]
        body_shape = motion_parms['betas'][frame_index]
        body_pose = motion_parms['pose_body'][frame_index]
        translation = motion_parms['trans'][frame_index]
        pose_params = torch.cat([global_orient.view(1, -1), body_pose.view(1, -1)], dim=1)
        shape_params = body_shape.view(1, -1)

        if cuda:
            pose_params = pose_params.cuda()
            shape_params = shape_params.cuda()
            translation = translation.cuda()

        verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)
        verts += translation.view(1, 1, -1)
        Jtr += translation.view(1, 1, -1)

        all_verts.append(verts.cpu().detach().numpy())
        all_jtr.append(Jtr.cpu().detach().numpy())

    all_verts = np.array(all_verts)
    all_jtr = np.array(all_jtr)

    # Calculate centroids for forehead and legs
    forehead_indices = lists['forehead']
    right_leg_indices = lists['right_leg']
    left_leg_indices = lists['left_leg']

    forehead_centroids = np.mean(all_verts[:, 0, forehead_indices], axis=1)
    right_leg_centroids = np.mean(all_verts[:, 0, right_leg_indices], axis=1)
    left_leg_centroids = np.mean(all_verts[:, 0, left_leg_indices], axis=1)

    # Calculate centroid of legs (average of right and left)
    leg_centroids = (right_leg_centroids + left_leg_centroids) / 2

    # Calculate Euclidean distance between forehead and leg centroids
    distances = np.linalg.norm(forehead_centroids - leg_centroids, axis=1)

    # Print results
    print(f"Distances for file {path}:")
    print(distances)

