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

# Define motion file path
motion_file_path = r'/media/lala/Crucial X62/CrosSim/Data/UniMocap/smplx_322/'
npy_files = glob.glob(os.path.join(motion_file_path, '**', '*.npy'), recursive=True)
absolute_paths = [os.path.abspath(file) for file in npy_files]

for path in absolute_paths:
    try:
        # Load motion file
        motion = np.load(path)
        motion = torch.tensor(motion).float()

        # Extract motion parameters
        motion_parms = {
            'root_orient': motion[:, :3],  
            'pose_body': motion[:, 3:3+63],  
            'pose_hand': motion[:, 66:66+90],  
            'pose_jaw': motion[:, 66+90:66+93],  
            'face_expr': motion[:, 159:159+50],  
            'face_shape': motion[:, 209:209+100],  
            'trans': motion[:, 309:309+3],  
            'betas': motion[:, 312:],  
        }

        # Add zeros for body pose
        zeros = torch.zeros(motion_parms['pose_body'].size(0), 6)
        motion_parms['pose_body'] = torch.cat((motion_parms['pose_body'], zeros), dim=1)

        # Initialize lists for results
        all_verts = []
        all_jtr = []
        all_poses = []  # Collect body poses
        all_trans = []  # Collect translations

        # Process each frame
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

            # Append data
            all_verts.append(verts.cpu().detach().numpy())
            all_jtr.append(Jtr.cpu().detach().numpy())
            all_poses.append(body_pose.cpu().detach().numpy())
            all_trans.append(translation.cpu().detach().numpy())

        # Save the data
        output_data = {
            'pose_body': np.array(all_poses),  # Body pose parameters
            'translation': np.array(all_trans),  # Translations
            'joints': np.array(all_jtr),  # Joint locations
        }

        save_path = os.path.join(
            '/media/lala/Crucial X62/CrosSim/Data/UniMocap/pose/', 
            f"{os.path.basename(path).split('.')[0]}_processed.npz"
        )
        np.savez(save_path, **output_data)
        print(f"Saved processed data to {save_path}")

    except Exception as e:
        print(f"Failed to process file: {path}. Error: {str(e)}")

