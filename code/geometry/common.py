import numpy as np


def rotation_error(R0, R1):
    return np.abs(
        np.arccos(np.clip((np.trace(R0.T @ R1) - 1) / 2.0, -0.999999,
                          0.999999))) / np.pi * 180


def translation_error(t0, t1):
    return np.linalg.norm(t0 - t1)


def angular_translation_error(t0, t1):
    t0 = t0 / np.linalg.norm(t0)
    t1 = t1 / np.linalg.norm(t1)
    err = np.arccos(np.clip(np.inner(t0, t1), -0.999999,
                            0.999999)) / np.pi * 180
    return err
