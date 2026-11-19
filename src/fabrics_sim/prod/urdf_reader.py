# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""Read a URDF into the kinematic arrays the fabric's kinematics kernels (prod/kinematics.py) need.

Upstream FABRICS imported the URDF into warp.sim's ModelBuilder, which current Warp no longer has. read_urdf reads it
with yourdfpy and returns the same arrays, named as on warp.sim's finalized Model, plus the joint-space maps
(UrdfInfo) the old importer returned. Joint order, transforms and axes follow the old warp.sim path exactly.
"""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import warp as wp
import yourdfpy

# Joint type codes, the values of the removed warp.sim constants.
JOINT_PRISMATIC = wp.constant(0)
JOINT_REVOLUTE = wp.constant(1)
JOINT_BALL = wp.constant(2)
JOINT_FIXED = wp.constant(3)
JOINT_FREE = wp.constant(4)


class UrdfInfo:
    """Maps between the robot's joint space (cspace: its revolute joints) and its links.

    Attributes:
        cspace_names: revolute joint names, in depth-first order from the base link.
        link_index_map: link name -> link index (the base link is 0).
        cspace2link: per cspace joint, the index of the link it moves.
        link2cspace: per link, the index of the last cspace joint at or above it (-1 for the base).
        cspace_joint_limits: per cspace joint, (lower, upper) in rad.
        cspace_name2index_map: joint name -> cspace index.
    """

    def __init__(self, cspace_names, link_index_map, cspace2link, link2cspace, cspace_joint_limits):
        self.cspace_names = cspace_names
        self.link_index_map = link_index_map
        self.cspace2link = cspace2link
        self.link2cspace = link2cspace
        self.cspace_joint_limits = cspace_joint_limits
        self.cspace_name2index_map = {name: i for i, name in enumerate(cspace_names)}


def matrix_to_rpy(R):
    """Roll, pitch and yaw (rad) of a 3x3 rotation matrix; urdfpy.utils.matrix_to_rpy's first solution."""
    R = np.asanyarray(R, dtype=np.float64)
    if np.abs(R[2, 0]) >= 1.0 - 1e-12:
        yaw = 0.0
        if R[2, 0] < 0:
            pitch = np.pi / 2
            roll = np.arctan2(R[0, 1], R[0, 2])
        else:
            pitch = -np.pi / 2
            roll = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        pitch = -np.arcsin(R[2, 0])
        roll = np.arctan2(R[2, 1] / np.cos(pitch), R[2, 2] / np.cos(pitch))
        yaw = np.arctan2(R[1, 0] / np.cos(pitch), R[0, 0] / np.cos(pitch))
    return np.array([roll, pitch, yaw], dtype=np.float64)


def append_subtree_joints(joints_by_parent, link, order):
    """Append the joints below link to order, depth first, branches in the order the URDF lists them."""
    for joint in joints_by_parent.get(link, []):
        order.append(joint)
        append_subtree_joints(joints_by_parent, joint.child, order)


def depth_first_joints(robot, base_link):
    """The robot's joints in depth-first order from base_link, as warp.sim's importer visited them."""
    joints_by_parent = {}
    for joint in robot.joints:
        joints_by_parent.setdefault(joint.parent, []).append(joint)

    order = []
    append_subtree_joints(joints_by_parent, base_link, order)
    return order


def read_urdf(filename, device, xform=None, verbose=False):
    """Read the kinematic tree of a fixed-base URDF with revolute and fixed joints.

    Args:
        filename: path of the URDF.
        device: Warp device for the arrays, e.g. "cuda:0".
        xform: wp.transform of the base in the world; identity if None.
        verbose: print the file being read.
    Returns:
        (info, model). info is a UrdfInfo. model holds Warp arrays named as on warp.sim's finalized Model:
        joint_type, joint_parent and joint_X_p for every joint (the base's fixed joint first), joint_axis for the
        revolute joints only, joint_q with one entry per revolute joint, and body_count.
    """
    if verbose:
        print("reading URDF:", filename)
    urdf = yourdfpy.URDF.load(filename, load_meshes=False, build_scene_graph=False, build_collision_scene_graph=False)
    xform = wp.transform_identity() if xform is None else xform

    children = {joint.child for joint in urdf.robot.joints}
    base_link = next(link.name for link in urdf.robot.links if link.name not in children)  # the link no joint moves

    cspace_names, cspace2link, link2cspace, cspace_joint_limits = [], [], [-1], []
    link_index = OrderedDict({base_link: 0})
    joint_type, joint_parent, joint_X_p, joint_axis = [int(JOINT_FIXED)], [-1], [xform], []

    for joint in depth_first_joints(urdf.robot, base_link):
        if joint.parent not in link_index:
            raise RuntimeError(f"Parent not found: {joint.parent}")

        origin = np.eye(4) if joint.origin is None else joint.origin
        rotation = wp.quat_rpy(*matrix_to_rpy(origin[:3, :3]))
        joint_X_p.append(wp.transform(origin[:3, 3], rotation))
        joint_parent.append(link_index[joint.parent])
        link = len(joint_type)

        if joint.type in ("revolute", "continuous"):
            lower, upper = -1.0e3, 1.0e3
            if joint.limit is not None and joint.limit.lower is not None:
                lower = joint.limit.lower
            if joint.limit is not None and joint.limit.upper is not None:
                upper = joint.limit.upper
            joint_type.append(int(JOINT_REVOLUTE))
            joint_axis.append(wp.normalize(wp.vec3(*joint.axis)))
            cspace_names.append(joint.name)
            cspace2link.append(link)
            cspace_joint_limits.append((lower, upper))
        elif joint.type == "fixed":
            joint_type.append(int(JOINT_FIXED))
        else:
            raise RuntimeError(f"Joint type {joint.type} of {joint.name} is not supported.")

        link2cspace.append(len(cspace_names) - 1)
        link_index[joint.child] = link

    model = SimpleNamespace(
        device=device,
        joint_type=wp.array(joint_type, dtype=int, device=device),
        joint_parent=wp.array(joint_parent, dtype=int, device=device),
        joint_X_p=wp.array(joint_X_p, dtype=wp.transform, device=device),
        joint_axis=wp.array(joint_axis, dtype=wp.vec3, device=device),
        joint_q=wp.zeros(len(cspace_names), dtype=float, device=device),
        body_count=len(joint_type),
    )
    info = UrdfInfo(cspace_names, link_index, cspace2link, link2cspace, cspace_joint_limits)
    return info, model
