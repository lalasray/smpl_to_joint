import os
import numpy as np
import torch
from smplpytorch.pytorch.smpl_layer import SMPL_Layer
from display_utils import display_model
import random
import matplotlib.pyplot as plt
import os
import glob

cuda = torch.cuda.is_available()
smpl_layer = SMPL_Layer(center_idx=0, gender='neutral', model_root='smplpytorch/native/models')
if cuda:
    smpl_layer = smpl_layer.cuda()

motion_file_path = r'/media/lala/Elements/motionx/motion_data/smplx_322'
npy_files = glob.glob(os.path.join(motion_file_path, '**', '*.npy'), recursive=True)
absolute_paths = [os.path.abspath(file) for file in npy_files]

for path in absolute_paths:
    try:
        motion_file_path = path
        motion = np.load(motion_file_path)
        motion = torch.tensor(motion).float()

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

    except Exception as e:
        print(f"Failed to process file: {motion_file_path}. Error: {str(e)}")
        continue  # Skip to the next file

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
        '''
        display_model(
            {'verts': verts, 'joints': Jtr},
            model_faces=smpl_layer.th_faces,
            with_joints=True,
            kintree_table=smpl_layer.kintree_table,
            savepath=f'/home/lala/Documents/GitHub/smpl_to_joint/pic/image_{frame_index}.png',
            show=False
        )
        '''
        all_verts.append(verts.cpu().detach().numpy())
        all_jtr.append(Jtr.cpu().detach().numpy())

    all_verts = np.array(all_verts)
    all_jtr = np.array(all_jtr)

    lists = {
        'right_wrist': [5405, 5430, 5431, 5567, 5569, 5667, 5668, 5669, 5670, 5696, 5702, 5705, 5740],
        'left_wrist': [1919, 1920, 1944, 1945, 1961, 1962, 1969, 1970, 2206, 2208, 2235, 2241, 2244],
        'right_shin': [4557, 4559, 4560, 4561, 4564, 4565, 4568, 4569, 4572, 4573, 4574, 4575, 4580, 4581, 4584, 4585, 4586, 4587, 4588, 4589, 4637, 4638, 4639, 4640, 4641, 4661, 4662, 4663, 4844, 4845, 4943, 4996],
        'left_shin': [1071, 1072, 1074, 1077, 1078, 1079, 1082, 1083, 1086, 1087, 1088, 1089, 1094, 1095, 1096, 1097, 1098, 1099, 1100, 1101, 1102, 1103, 1151, 1152, 1153, 1154, 1155, 1175, 1176, 1177, 1371, 1372],
        'right_thigh': [847, 848, 849, 850, 872, 873, 874, 875, 876, 877, 904, 905, 906, 907, 957, 1159, 1365, 1366, 1499, 1500],
        'left_thigh': [4333, 4334, 4335, 4336, 4358, 4359, 4360, 4361, 4362, 4363, 4645, 4648, 4711, 4712, 4801, 4802, 4839],
        'right_arm': [1546, 1547, 1548, 1549, 1550, 1551, 1556, 1557, 1568, 1569, 1570, 1571, 1589, 1590, 1591, 1597, 1598, 1599, 1600, 1601, 1602, 1604, 1685, 1686, 1693, 1913, 1976, 1980],
        'left_arm': [5015, 5016, 5017, 5018, 5019, 5020, 5025, 5027, 5028, 5039, 5040, 5058, 5059, 5060, 5061, 5067, 5068, 5069, 5070, 5071, 5160, 5162, 5210, 5407, 5408, 5409, 5412, 5426],
        'right_shoulder': [788, 789, 1310, 1311, 1315, 1378, 1379, 1405, 1406, 1407, 1505, 1506, 1542, 2821, 2822, 2895, 2896],
        'left_shoulder': [4790, 4791, 4794, 4795, 4849, 4850, 4851, 4852, 5011, 5148, 5149, 5151, 5181, 5185],
        'forehead': [0, 1, 5, 132, 133, 232, 234, 235, 259, 335, 336, 3512, 3513, 3514, 3515, 3517, 3644, 3645, 3646, 3676, 3744, 3745, 3746, 3771],
        'right_foot': [3327, 3333, 3334, 3335, 3337, 3338, 3340, 3341, 3342, 3343, 3344, 3345, 3347, 3364, 3365, 3366, 3367, 3368, 3370, 3371, 3374, 3379, 3380, 3399, 3400, 3401, 3469],
        'left_foot': [6692, 6703, 6704, 6713, 6728, 6734, 6735, 6740, 6741, 6742, 6743, 6744, 6745, 6765, 6766, 6767, 6768, 6770, 6779, 6780, 6799, 6800, 6869],
        'back': [726, 727, 731, 732, 745, 746, 748, 811, 895, 896, 1213, 1214, 1215, 1216, 1217, 1218, 1220, 1301, 1302, 1303, 1304, 1305, 1306, 1753, 1754, 1755, 1820, 2875, 2876, 2877, 2878, 2882, 2884, 2885, 2973, 3012, 3471, 3482, 4214, 4217, 4299, 4696, 4697, 4698, 4699, 4700, 4701, 4703, 4783, 4784, 4785, 4786, 5222, 6336, 6337, 6343],
        'right_shirt_pocket': [598, 599, 600, 601, 652, 670, 684, 685, 686, 687, 691, 942, 943, 1254, 1255, 1256, 1257, 1349, 1350, 1351, 1352, 2852, 2853, 2854, 2856, 2857, 2858, 3030, 3031, 3032, 3033, 3040, 3042, 3483],
        'left_shirt_pocket': [4086, 4087, 4088, 4089, 4141, 4156, 4157, 4158, 4159, 4172, 4173, 4174, 4175, 4180, 4428, 4429, 4679, 4739, 4740, 4825, 4826, 4827, 4828, 4893, 4894, 6315, 6317, 6318, 6319, 6320, 6321, 6477],
        'chest': [1200, 1257, 1329, 1348, 2870, 2871, 3063, 3076, 3077, 3079, 3506, 4179, 4688, 4737, 4738, 4824, 6331, 6332, 6498],
        'Necklace': [1427, 2872, 3061, 3062, 3067, 3168, 3169, 3171, 4187, 4782, 4900, 6333, 6496, 6497, 6502],
        'belt': [2922, 2923, 3152, 3153, 3160, 3507, 6381, 6382, 6568, 6569],
        'left_ear': [3990, 3992, 3993, 3995, 3996, 3997, 3998, 3999, 4000, 4001, 4002, 4003, 4004, 4016, 4023, 4024, 4025, 4026, 4027, 4028, 4029, 4031, 4032, 4033, 4034, 4035, 4041, 4042, 4046, 4048, 4066, 4067, 4068, 4069, 4070, 4071],
        'right_ear': [449, 466, 502, 505, 506, 507, 508, 509, 510, 511, 513, 514, 515, 516, 535, 537, 538, 539, 541, 543, 546, 554, 560, 561, 578, 579, 580, 581, 582, 583, 1770, 1773, 1774, 3485, 3486, 3487, 3488, 3489, 3491, 3493, 3494]
    }

    num_samples = 3
    selected_samples = {}
    for key, lst in lists.items():
        selected_samples[key] = random.sample(lst, num_samples)

    '''
    for key, sampled_list in selected_samples.items():
        print(f"Selected vertices for {key}:", sampled_list)
    '''

    def calculate_linear_acceleration_and_angular_velocity(all_verts, selected_vertices):
        num_frames = all_verts.shape[0]

        centroids = np.zeros((num_frames, 3))
        normals = np.zeros((num_frames, 3))

        for frame in range(num_frames):
            v1 = all_verts[frame, 0, selected_vertices[0], :]
            v2 = all_verts[frame, 0, selected_vertices[1], :]
            v3 = all_verts[frame, 0, selected_vertices[2], :]

            centroids[frame, :] = (v1 + v2 + v3) / 3.0
            normal_vector = np.cross(v2 - v1, v3 - v1)
            normals[frame, :] = normal_vector / np.linalg.norm(normal_vector)  

        velocities = np.diff(centroids, axis=0)  
        linear_acceleration = np.diff(velocities, axis=0) 
        normal_diff = np.diff(normals, axis=0) 
        angular_velocity = np.cross(normal_diff, normals[1:, :])

        return centroids,normals, linear_acceleration, angular_velocity

    #body_part = 'left_wrist'
    #selected_vertices = selected_samples[body_part]
    #positions, orientations, linear_accel, angular_vel = calculate_linear_acceleration_and_angular_velocity(all_verts, selected_vertices)

    '''
    print(f"Linear acceleration for {body_part}:\n", linear_accel.shape)
    print(f"Angular velocity for {body_part}:\n", angular_vel.shape)

    time_accel = np.arange(linear_accel.shape[0])
    time_angular = np.arange(angular_vel.shape[0])

    plt.figure(figsize=(12, 6))
    plt.subplot(2, 1, 1)
    plt.plot(time_accel, linear_accel[:, 0], label='X-axis')
    plt.plot(time_accel, linear_accel[:, 1], label='Y-axis')
    plt.plot(time_accel, linear_accel[:, 2], label='Z-axis')
    plt.title(f'Linear Acceleration for {body_part}')
    plt.xlabel('Time Frame')
    plt.ylabel('Linear Acceleration')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(time_angular, angular_vel[:, 0], label='X-axis')
    plt.plot(time_angular, angular_vel[:, 1], label='Y-axis')
    plt.plot(time_angular, angular_vel[:, 2], label='Z-axis')
    plt.title(f'Angular Velocity for {body_part}')
    plt.xlabel('Time Frame')
    plt.ylabel('Angular Velocity')
    plt.legend()

    plt.tight_layout()
    plt.show()
    '''

    base_dir, base_filename = os.path.split(motion_file_path)
    base_name = os.path.splitext(base_filename)[0]
    save_dir = os.path.join(base_dir, base_name)
    os.makedirs(save_dir, exist_ok=True)
    for body_part, selected_vertices in selected_samples.items():
        positions, orientations, linear_accel, angular_vel = calculate_linear_acceleration_and_angular_velocity(all_verts, selected_vertices)
        file_path = os.path.join(save_dir, f'{body_part}_v2.npz')
        np.savez(file_path,
                positions=positions,
                orientations=orientations,
                linear_acceleration=linear_accel,
                angular_velocity=angular_vel)
        
        print(f"Data saved successfully for {body_part} at {file_path}")