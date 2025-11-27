import numpy as np
from . import utils as feat_utils
import torch

def kinetic_features(joints, frame_time=1/60., up_vec="y", sliding_window=2):

    b, t, j, _ = joints.shape
    joints = joints.numpy()

    def window_velocity(seq, window_size):
        """
        seq: (t, j, 3)
        return: (t, j, 3)
        """
        v = np.zeros_like(seq)
        for i in range(1, t):  
            cur_v = []
            for k in range(-window_size, window_size+1):
                if 0 <= i+k < t:
                    cur_v.append((seq[i+k] - seq[i+k-1]) / frame_time)
            v[i] = np.mean(cur_v, axis=0) if len(cur_v) > 0 else 0
        return v

    # -------- velocity for all batches --------
    velocities = np.zeros((b, t, j, 3))
    for bi in range(b):
        velocities[bi] = window_velocity(joints[bi], sliding_window)

    # -------- horizontal & vertical --------
    if up_vec == "y":
        horizontal_velocities = velocities[:, :, :, [0, 2]]
        vertical_velocities   = velocities[:, :, :, 1:2]
    elif up_vec == "z":
        horizontal_velocities = velocities[:, :, :, [0, 1]]
        vertical_velocities   = velocities[:, :, :, 2:3]
    else:
        raise NotImplementedError("up_vec must be 'y' or 'z'.")

    # -------- kinetic energy --------
    kinetic_energy_h = np.mean(np.linalg.norm(horizontal_velocities, axis=-1)**2, axis=1)  # (b, j)
    kinetic_energy_v = np.mean(np.linalg.norm(vertical_velocities, axis=-1)**2, axis=1)    # (b, j)

    # -------- acceleration --------
    def window_acceleration(seq, window_size):
        """
        seq: (t, j, 3)
        return: (t, j, 3)
        """
        a = np.zeros_like(seq)
        v = window_velocity(seq, window_size)
        for i in range(1, t):
            cur_a = []
            for k in range(-window_size, window_size+1):
                if 0 <= i+k < t:
                    cur_a.append((v[i+k] - v[i+k-1]) / frame_time)
            a[i] = np.mean(cur_a, axis=0) if len(cur_a) > 0 else 0
        return a

    accelerations = np.zeros((b, t, j, 3))
    for bi in range(b):
        accelerations[bi] = window_acceleration(joints[bi], sliding_window)

    energy_expenditure = np.mean(np.linalg.norm(accelerations, axis=-1), axis=1)  # (b, j)

    # -------- stack features --------
    kinetic_feats = []
    for i in range(j):
        feat = np.stack([
            kinetic_energy_h[:, i],
            kinetic_energy_v[:, i],
            energy_expenditure[:, i]
        ], axis=-1)  # (b, 3)
        kinetic_feats.append(feat)

    kinetic_feats = np.concatenate(kinetic_feats, axis=-1)  # (b, 72)
    kinetic_feats = torch.from_numpy(kinetic_feats.astype(np.float32))
    return kinetic_feats[:, :66]

def extract_kinetic_features(positions):
    assert len(positions.shape) == 3  # (seq_len, n_joints, 3)
    features = KineticFeatures(positions)
    kinetic_feature_vector = []
    for i in range(positions.shape[1]):
        feature_vector = np.hstack(
            [
                features.average_kinetic_energy_horizontal(i),
                features.average_kinetic_energy_vertical(i),
                features.average_energy_expenditure(i),
            ]
        )
        kinetic_feature_vector.extend(feature_vector)
    kinetic_feature_vector = np.array(kinetic_feature_vector, dtype=np.float32)
    return kinetic_feature_vector


class KineticFeatures:
    def __init__(
        self, positions, frame_time=1./60, up_vec="y", sliding_window=2
    ):
        self.positions = positions
        self.frame_time = frame_time
        self.up_vec = up_vec
        self.sliding_window = sliding_window

    def average_kinetic_energy(self, joint):
        average_kinetic_energy = 0
        for i in range(1, len(self.positions)):
            average_velocity = feat_utils.calc_average_velocity(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
            average_kinetic_energy += average_velocity ** 2
        average_kinetic_energy = average_kinetic_energy / (
            len(self.positions) - 1.0
        )
        return average_kinetic_energy

    def average_kinetic_energy_horizontal(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = feat_utils.calc_average_velocity_horizontal(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_kinetic_energy_vertical(self, joint):
        val = 0
        for i in range(1, len(self.positions)):
            average_velocity = feat_utils.calc_average_velocity_vertical(
                self.positions,
                i,
                joint,
                self.sliding_window,
                self.frame_time,
                self.up_vec,
            )
            val += average_velocity ** 2
        val = val / (len(self.positions) - 1.0)
        return val

    def average_energy_expenditure(self, joint):
        val = 0.0
        for i in range(1, len(self.positions)):
            val += feat_utils.calc_average_acceleration(
                self.positions, i, joint, self.sliding_window, self.frame_time
            )
        val = val / (len(self.positions) - 1.0)
        return val
