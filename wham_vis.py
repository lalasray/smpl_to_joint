import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from display_utils import display_model
import os

def modify_poses(pose_params):
    """Modify standing poses and arm movements to appear less static"""
    modified_poses = pose_params.clone()
    for i in range(modified_poses.shape[0]):
        pose = modified_poses[i]
        
        # Add slight knee bend if standing (prevent perfect straightness)
        if torch.abs(pose[1]) + torch.abs(pose[4]) < 0.1:
            pose[1] += 0.7  # Left knee
            pose[4] += 0.7  # Right knee
        
        # Reduce exaggerated arm movement
        pose[16:22] *= 0.2  # Scale down arm joints motion
    
    return modified_poses

def smooth_poses(pose_params, alpha=0.3):
    """Smooth pose transitions using an exponential moving average filter"""
    smoothed_poses = pose_params.clone()
    for i in range(1, smoothed_poses.shape[0]):
        smoothed_poses[i] = alpha * smoothed_poses[i] + (1 - alpha) * smoothed_poses[i - 1]
    return smoothed_poses

def generate_smpl_video(input_file, output_video='output.mp4', fps=30):
    # Load data
    loaded_data = np.load(input_file)
    print("Keys in loaded data:", loaded_data.files)
    
    pose_params = torch.tensor(loaded_data["pose"][::10], dtype=torch.float32)  # All frames downsampled
    #pose_params = torch.tensor(loaded_data["pose"], dtype=torch.float32)  # All frames
    shape_params = torch.tensor(loaded_data["betas"][:1], dtype=torch.float32)  # First frame only
    pose_params = smooth_poses(pose_params) 
    #pose_params[48:72] = 0
    
    print(pose_params.shape) 
    # Use GPU if available
    cuda = torch.cuda.is_available()
    if cuda:
        pose_params = pose_params.cuda()
        shape_params = shape_params.cuda()
    
    # Create SMPL layer
    smpl_layer = SMPL_Layer(center_idx=0, gender='neutral', model_root='smplpytorch/native/models')
    if cuda:
        smpl_layer.cuda()
    
    frame_height, frame_width = 500, 500
    video_writer = cv2.VideoWriter(output_video, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    
    for i in range(pose_params.shape[0]):
        verts, Jtr = smpl_layer(pose_params[i:i+1], th_betas=shape_params)
        verts[:, :, 1] *= -1  # Invert Y-axis
        Jtr[:, :, 1] *= -1
        
        frame_path = f'frame_{i}.png'
        display_model({'verts': verts.cpu().detach(), 'joints': Jtr.cpu().detach()},
                      model_faces=smpl_layer.th_faces,
                      with_joints=True,
                      kintree_table=smpl_layer.kintree_table,
                      savepath=frame_path,
                      show=False)
        
        frame = cv2.imread(frame_path)
        frame = cv2.resize(frame, (frame_width, frame_height))
        video_writer.write(frame)
        plt.close('all') 
        os.remove(frame_path)  # Clean up
    
    video_writer.release()
    print(f"Video saved as {output_video}")

def generate_smpl_images(input_file, output_folder='output_frames'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    loaded_data = np.load(input_file)
    print("Keys in loaded data:", loaded_data.files)

    pose_params = torch.tensor(loaded_data["pose"][::10], dtype=torch.float32)  # Downsampled frames
    shape_params = torch.tensor(loaded_data["betas"][:1], dtype=torch.float32)  # First frame only
    pose_params[:, 54:72] = pose_params[:, 54:72]*0.5
    pose_params[:, 36:38] = pose_params[:, 36:38]*0
    pose_params[:, 4] = pose_params[:, 4]+0.5
    pose_params[:, 7] = pose_params[:, 7]+0.5
    pose_params[:, 10] = pose_params[:, 10]+0.5
    
    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_params = pose_params.to(device)
    shape_params = shape_params.to(device)

    # Create SMPL layer
    smpl_layer = SMPL_Layer(center_idx=0, gender='neutral', model_root='smplpytorch/native/models').to(device)

    for i in range(pose_params.shape[0]):
        verts, Jtr = smpl_layer(pose_params[i:i+1], th_betas=shape_params)
        verts[:, :, 1] *= -1  # Invert Y-axis
        Jtr[:, :, 1] *= -1

        frame_path = os.path.join(output_folder, f'frame_{i:04d}.png')  # Save with zero-padded numbering
        display_model({'verts': verts.cpu().detach(), 'joints': Jtr.cpu().detach()},
                      model_faces=smpl_layer.th_faces,
                      with_joints=True,
                      kintree_table=smpl_layer.kintree_table,
                      savepath=frame_path,
                      show=False)

        plt.close('all')  # Close figure to prevent memory leak

    print(f"All frames saved in '{output_folder}'.")

import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os

def generate_com_images(input_file, output_folder='CoM'):
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    # Load data
    loaded_data = np.load(input_file)
    print("Keys in loaded data:", loaded_data.files)

    pose_params = torch.tensor(loaded_data["pose"][::10], dtype=torch.float32)  # Downsampled frames
    shape_params = torch.tensor(loaded_data["betas"][:1], dtype=torch.float32)  # First frame only
    new_pose_params = pose_params
    new_pose_params[:, 54:72] = pose_params[:, 54:72]*0.5
    new_pose_params[:, 36:38] = pose_params[:, 36:38]*0
    new_pose_params[:, 4] = pose_params[:, 4]+0.5
    new_pose_params[:, 7] = pose_params[:, 7]+0.5
    new_pose_params[:, 10] = pose_params[:, 10]+0.5

    # Use GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pose_params = pose_params.to(device)
    shape_params = shape_params.to(device)
    new_pose_params = new_pose_params.to(device)
    

    # Create SMPL layer
    smpl_layer = SMPL_Layer(center_idx=0, gender='neutral', model_root='smplpytorch/native/models').to(device)

    for i in range(pose_params.shape[0]):
        verts, Jtr = smpl_layer(pose_params[i:i+1], th_betas=shape_params)
        verts[:, :, 1] *= -1  # Invert Y-axis
        Jtr[:, :, 1] *= -1

        newverts, newJtr = smpl_layer(new_pose_params[i:i+1], th_betas=shape_params)
        newverts[:, :, 1] *= -1  # Invert Y-axis
        
        # Calculate the GT CoM (center of mass from vertices)
        com = verts.mean(dim=1).squeeze()  # Mean over the vertices (axis 1: x, y, z)

        newcom = newverts.mean(dim=1).squeeze()

        # The root joint is typically the first joint in the joint hierarchy (index 0)
        pose_com = Jtr[0, 0, :]  # Pose CoM (root joint position)

        # Create a figure and 3D axis
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')

        # Plot the GT CoM (center of mass from vertices) as a red sphere
        ax.scatter(newcom[0].cpu().detach().numpy(),
                   newcom[1].cpu().detach().numpy(),
                   newcom[2].cpu().detach().numpy(),
                   color='green', s=100, label='GT CoM', marker='o')

        # Plot the GT CoM (center of mass from vertices) as a red sphere
        ax.scatter(com[0].cpu().detach().numpy(),
                   com[1].cpu().detach().numpy(),
                   com[2].cpu().detach().numpy(),
                   color='red', s=100, label='Pose based CoM', marker='o')

        # Plot the Pose CoM (root joint) as a blue sphere
        ax.scatter(pose_com[0].cpu().detach().numpy(),
                   pose_com[1].cpu().detach().numpy(),
                   pose_com[2].cpu().detach().numpy(),
                   color='blue', s=100, label='Pressure based CoM', marker='o')

        # Add labels and show the plot
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')
        ax.legend()

        # Save the frame with the two center of mass markers
        frame_path = os.path.join(output_folder, f'frame_{i:04d}.png')
        plt.savefig(frame_path)
        plt.close('all')  # Close figure to prevent memory leak

    print(f"All frames saved in '{output_folder}'.")


# Example usage
#generate_smpl_images(r"C:\Users\lalas\Downloads\Lars.npz")
generate_com_images(r"C:\Users\lalas\Downloads\Lars.npz")
