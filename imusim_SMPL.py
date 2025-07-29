import os
import numpy as np
from scipy.interpolate import interp1d
from imusim.all import *
import matplotlib.pyplot as plt

class MyTrajectory:
    def __init__(self, positions, orientations, vel, acc, a_vel, a_acc, sf):
        self.sf = sf  # sampling frequency, e.g. 15 Hz
        self.time = np.arange(len(positions)) / sf

        self.startTime = self.time[0]
        self.endTime = self.time[-1]

        # Build interpolators for position, velocity, acceleration
        self.pos_interp = interp1d(self.time, positions, axis=0, fill_value="extrapolate")
        self.vel_interp = interp1d(self.time, vel, axis=0, fill_value="extrapolate")
        self.acc_interp = interp1d(self.time, acc, axis=0, fill_value="extrapolate")

        # Store quaternions in (w, x, y, z) format for IMUSim
        self.quaternions = np.column_stack([orientations[:, 3], orientations[:, :3]])

        self.omega_interp = interp1d(self.time, a_vel, axis=0, fill_value="extrapolate")
        self.alpha_interp = interp1d(self.time, a_acc, axis=0, fill_value="extrapolate")

    def position(self, t):
        p = self.pos_interp(t)
        return vector(*p)

    def velocity(self, t):
        v = self.vel_interp(t)
        return vector(*v)

    def acceleration(self, t):
        a = self.acc_interp(t)
        return vector(*a)

    def rotation(self, t):
        # Use nearest quaternion
        idx = np.clip(np.searchsorted(self.time, t) - 1, 0, len(self.time) - 1)
        q = self.quaternions[idx]
        return Quaternion(q[0], q[1], q[2], q[3])

    def rotationalVelocity(self, t):
        w = self.omega_interp(t)
        return vector(*w)

    def rotationalAcceleration(self, t):
        a = self.alpha_interp(t)
        return vector(*a)

def simulate_imu(path, sf, plot_data=False):
    data = np.load(path)

    positions = data['positions']
    orientations = data['orientations']
    vel = data['linear_velocity']
    acc = data['global_acceleration']
    a_vel = data['angular_velocity_world']
    a_acc = data['angular_acceleration']

    # Create your trajectory
    traj = MyTrajectory(positions, orientations, vel, acc, a_vel, a_acc, sf)

    sim = Simulation()
    #imu = Orient3IMU(sim, traj)
    imu = IdealIMU(sim, traj)
    behaviour = BasicIMUBehaviour(imu, 1.0 / sf)

    sim.time = traj.startTime
    print("Running simulation...")
    sim.run(traj.endTime)
    print("Done!")

    # Extract numpy arrays from TimeSeries vectors
    acc_data = np.array([imu.accelerometer.rawMeasurements.values.squeeze().T])
    gyro_data = np.array([imu.gyroscope.rawMeasurements.values.squeeze().T])

    if plot_data:
        plt.figure()
        plot(imu.accelerometer.rawMeasurements)
        plt.title("Synthetic Accelerometer for right_wrist")
        plt.show()

        plot(imu.gyroscope.rawMeasurements)
        plt.title("Synthetic Gyroscope for right_wrist")
        plt.show()

    return acc_data, gyro_data



def process_directory(root_dir, sf):
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith("sIMU.npz"):
                new_filename = filename.replace("sIMU", "imusim")
                new_path = os.path.join(dirpath, new_filename)

                # Skip if output file already exists
                if os.path.exists(new_path):
                    print(f"Skipping (already processed): {new_path}")
                    continue

                full_path = os.path.join(dirpath, filename)
                print(f"Processing: {full_path}")

                try:
                    acc_data, gyro_data = simulate_imu(full_path, sf, plot_data=False)

                    np.savez(new_path, accelerometer=acc_data, gyroscope=gyro_data)
                    print(f"Saved: {new_path}\n")

                except Exception as e:
                    print(f"Failed to process {full_path}: {e}\n")



# ===== Example usage =====
process_directory("/home/lala/Documents/Data/VQIMU/UTD_MHAD", sf=15)
