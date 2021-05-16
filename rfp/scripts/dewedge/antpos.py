"""Different tools for creating arrays."""

import numpy as np
from scipy.spatial.transform import Rotation

# See https://en.wikipedia.org/wiki/Golomb_ruler
GOLOMB_RULERS = {
    12 : [0, 2, 6, 24, 29, 40, 43, 55, 68, 75, 76, 85],
    13 : [0, 2, 5, 25, 37, 43, 59, 70, 85, 89, 98, 99, 106],
    14 : [0, 4, 6, 20, 35, 52, 59, 77, 78, 86, 89, 99, 122, 127],
    15 : [0, 4, 20, 30, 57, 59, 62, 76, 100, 111, 123, 136,
          144, 145, 151],
    16 : [0, 1, 4, 11, 26, 32, 56, 68, 76, 115, 117, 134, 150,
          163, 168, 177],
    17 : [0, 5, 7, 17, 52, 56, 67, 80, 81, 100, 122, 138, 159,
          165, 168, 191, 199],
    18 : [0, 2, 10, 22, 53, 56, 82, 83, 89, 98, 130, 148, 153,
          167, 188, 192, 205, 216],
    19 : [0, 1, 6, 25, 32, 72, 100, 108, 120, 130, 153, 169, 187,
          190, 204, 231, 233, 242, 246],
    20 : [0, 1, 8, 11, 68, 77, 94, 116, 121, 156, 158, 179, 194,
          208, 212, 228, 240, 253, 259, 283],
    21 : [0, 2, 24, 56, 77, 82, 83, 95, 129, 144, 179, 186, 195,
          255, 265, 285, 293, 296, 310, 329, 333],
    22 : [0, 1, 9, 14, 43, 70, 106, 122, 124, 128, 159, 179, 204,
          223, 253, 263, 270, 291, 330, 341, 353, 356],
    23 : [0, 3, 7, 17, 61, 66, 91, 99, 114, 159, 171, 199, 200, 226,
          235, 246, 277, 316, 329, 348, 350, 366, 372],
    24 : [0, 9, 33, 37, 38, 97, 122, 129, 140, 142, 152, 191, 205,
          208, 252, 278, 286, 326, 332, 353, 368, 384, 403, 425],
    25 : [0, 12, 29, 39, 72, 91, 146, 157, 160, 161, 166, 191, 207,
          214, 258, 290, 316, 354, 372, 394, 396, 431, 459, 467, 480],
    26 : [0, 1, 33, 83, 104, 110, 124, 163, 185, 200, 203, 249, 251,
          258, 314, 318, 343, 356, 386, 430, 440, 456, 464, 475, 487, 492],
    27 : [0, 3, 15, 41, 66, 95, 97, 106, 142, 152, 220, 221, 225, 242, 295,
          330, 338, 354, 382, 388, 402, 415, 486, 504, 523, 546, 553]
}

def golomb_ruler(order):
    """Retrieve the markings on a Golomb ruler of a given order."""
    return np.array(GOLOMB_RULERS[order])

def golomb_array(order=20, base_sep=14.6, angle=0):
    """
    Construct a linear array with spacings based on a Golomb ruler.

    Parameters
    ----------
    order: int, optional
        Order of the Golomb ruler to use (i.e. number of antennas).
        Default is to make an array with 20 antennas.
    base_sep: float, optional
        Length of the shortest baseline, in meters. Default is 14.6 meters.
    axis: float, optional
        Orientation of the ruler in a locally horizontal plane, measured in
        radians clockwise from East. Default is to make an East-West array.

    Returns
    -------
    antpos: dict
        Dictionary mapping antenna numbers to ENU positions.
    """
    rot_mat = planar_rotation(angle, direction="u")
    ruler = base_sep * golomb_ruler(order)
    return {ant: rot_mat @ [pos, 0, 0] for ant, pos in enumerate(ruler)}


def linear_array(Nants=20, base_sep=14.6, angle=0):
    """
    Construct a linear array with uniform spacing between antennas.

    Parameters
    ----------
    Nants: int, optional
        Number of antennas in the array. Default is 20 antennas.
    base_sep: float, optional
        Length of the shortest baseline, in meters. Default is 14.6 meters.
    axis: float, optional
        Orientation of the ruler in a locally horizontal plane, measured in
        radians clockwise from East. Default is to make an East-West array.

    Returns
    -------
    antpos: dict
        Dictionary mapping antenna numbers to ENU positions.
    """
    rot_mat = planar_rotation(angle, direction="u")
    return {ant: rot_mat @ [ant * base_sep, 0, 0] for ant in range(Nants)}


def planar_rotation(angle, direction=2):
    """Calculate a rotation in the plane."""
    if isinstance(direction, str):
        if len(direction) != 1:
            raise NotImplementedError(
                "Only rotations about a single axis are currently supported."
            )
        if direction.lower() in "enu":
            direction = "enu".index(direction.lower())
        elif direction.lower() in "xyz":
            direction = "xyz".index(direction.lower())
        else:
            raise NotImplementedError("Unsupported direction string.")
    
    rotvec = np.where(np.arange(3) == direction, 1, 0)
    rotation = Rotation.from_rotvec(angle * rotvec)
    return rotation.to_dcm()
