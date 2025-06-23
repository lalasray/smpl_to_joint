import torch
import numpy as np
import time
import matplotlib.pyplot as plt
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from display_utils import display_model
import pickle
import joblib
import os
import re
import argparse



def args_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',   type = str)
    parser.add_argument('--output_path', type = str)
    parser.add_argument('--dataset',     type = str)
    parser.add_argument('--device',      type = str)

    args = parser.parse_args()
    return args




def _load_smpl_data(args):
    final_path = os.path.join(args.data_path, 'wham_output.pkl')
    print(final_path)
    smpl_data = joblib.load(final_path)
    print(smpl_data[0])
    return smpl_data[0]




if __name__ == '__main__':
    args = args_parse()

    loaded_data = _load_smpl_data(args=args)
    savepath    = os.path.join(args.output_path,'frames')
    os.makedirs(savepath, exist_ok = True)


    poses = loaded_data['pose']  
    betas = loaded_data['betas']  
    trans = torch.Tensor(loaded_data["trans"]).cuda()

    print('Pose shape:',  poses.shape)
    print('Betas shape:', betas.shape)

    smpl_layer = SMPL_Layer(
        center_idx=0,
        gender='neutral',
        model_root='/home/calatrava/Documents/PhD/Thesis/Experiments/PerCom2026/smpl_to_joint/smplpytorch/native/models'
    )

    smpl_layer.cuda()
    num_frames = poses.shape[0]
    print("Hi!")
    print(num_frames)

    for i in range(num_frames):
        # Single frame pose & betas
        pose_params  = torch.tensor(poses[i:i+1], dtype=torch.float32)
        shape_params = torch.tensor(betas[i:i+1], dtype=torch.float32)

        pose_params  = pose_params.cuda()
        shape_params = shape_params.cuda()

        with torch.no_grad():
            verts, Jtr = smpl_layer(pose_params, th_betas=shape_params)

        # Optional: flip Y axis for your coordinate system
        verts[:, :, 1] *= -1
        Jtr[:, :, 1]   *= -1
        
        # add translation
        verts = verts + trans[:, None, :] 
        Jtr   = Jtr   + trans[:, None, :]

        savename = f'frame_{i:04d}.png'
        final_savepath = os.path.join(savepath, savename)

        # Save to disk
        display_model(
            {'verts': verts.detach().cpu(),
            'joints': Jtr.detach().cpu()},
            model_faces=smpl_layer.th_faces.cpu(),
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath=final_savepath,
            show=False
        )

        print(f'Saved: {savepath}')
