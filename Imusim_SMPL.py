import numpy as np
from scipy.interpolate import interp1d
from imusim.all import *

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

import numpy as np

# Load your saved data
data = np.load("/home/lala/2/right_wrist.npz")

sf = 15  # your sampling frequency
positions = data['positions']
orientations = data['orientations']
vel = data['linear_velocity']
acc = data['global_acceleration']
a_vel = data['angular_velocity_world']
a_acc = data['angular_acceleration']

# Create your trajectory
traj = MyTrajectory(positions, orientations, vel, acc, a_vel, a_acc, sf)

from imusim.all import *

sim = Simulation()
#imu = Orient3IMU(sim, traj)
imu = IdealIMU(sim, traj)
behaviour = BasicIMUBehaviour(imu, 1.0 / sf)

sim.time = traj.startTime
print("Running simulation...")
sim.run(traj.endTime)
print("Done!")

import matplotlib.pyplot as plt
plt.figure()
plot(imu.accelerometer.rawMeasurements)
plt.title("Synthetic Accelerometer for right_wrist")
plt.show()
