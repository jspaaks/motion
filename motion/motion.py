import numpy
from scipy.spatial.transform import Rotation
from typing import Optional


def translational_motion_1d(time: numpy.ndarray, position: numpy.ndarray, v0=0.0):

    velocity = numpy.full(position.shape, numpy.nan)
    acceleration = numpy.full(position.shape, numpy.nan)
    for i, _ in enumerate(time):
        if i == 0:
            velocity[i] = v0
        else:
            delta_position = position[i] - position[i - 1]
            delta_time = time[i] - time[i - 1]
            acceleration[i] = 2 * (delta_position - velocity[i - 1] * delta_time) / numpy.power(delta_time, 2)
            velocity[i] = velocity[i - 1] + acceleration[i] * delta_time
    return acceleration


def motion_6d(t: numpy.ndarray, x: numpy.ndarray, y: numpy.ndarray, z: numpy.ndarray,
              rx: Optional[numpy.ndarray] = None, ry: Optional[numpy.ndarray] = None,
              rz: Optional[numpy.ndarray] = None, vx0=0.0, vy0=0.0, vz0=0.0, g=-9.81) -> numpy.ndarray:

    if rx is None:
        rx = numpy.zeros_like(t)

    if ry is None:
        ry = numpy.zeros_like(t)

    if rz is None:
        rz = numpy.zeros_like(t)

    rotation_vectors = numpy.stack((rx, ry, rz), axis=0).T
    unit_vector = [0, 0, 1]
    orientation_x, orientation_y, orientation_z = Rotation.from_rotvec(rotation_vectors).apply(unit_vector).T

    acceleration_x = translational_motion_1d(t, position=x, v0=vx0)
    acceleration_y = translational_motion_1d(t, position=y, v0=vy0)
    acceleration_z = translational_motion_1d(t, position=z, v0=vz0)

    gx = acceleration_x / g
    gy = acceleration_y / g
    gz = (acceleration_z - g) / g

    return numpy.stack((orientation_x, orientation_y, orientation_z, gx, gy, gz), axis=0)
