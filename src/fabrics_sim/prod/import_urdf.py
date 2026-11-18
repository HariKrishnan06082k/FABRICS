# Copyright (c) 2022 NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

""" URDF import module.

Reads a URDF with yourdfpy and returns the kinematic arrays the fabric kinematics kernels need: the arrays that
warp.sim's ModelBuilder used to provide (warp.sim is gone from current Warp), plus the cspace maps that
parse_urdf_annotated always returned. Joint order, transforms and axes follow the old warp.sim path exactly.
"""

from collections import OrderedDict
from types import SimpleNamespace

import numpy as np
import warp as wp
import yourdfpy

# Joint type codes, same values as the removed warp.sim constants.
JOINT_PRISMATIC = wp.constant(0)
JOINT_REVOLUTE = wp.constant(1)
JOINT_BALL = wp.constant(2)
JOINT_FIXED = wp.constant(3)
JOINT_FREE = wp.constant(4)


class UrdfInfo:
    def __init__(self, cspace_names, link_index_map, cspace2link, link2cspace, cspace_joint_limits):
        self.cspace_names = cspace_names
        self.link_index_map = link_index_map
        self.cspace2link = cspace2link
        self.link2cspace = link2cspace
        self.cspace_joint_limits = cspace_joint_limits

        self.cspace_name2index_map = {}
        for i, name in enumerate(self.cspace_names):
            self.cspace_name2index_map[name] = i


def matrix_to_rpy(R):
    """Roll-pitch-yaw of a 3x3 rotation matrix, copied from urdfpy.utils.matrix_to_rpy (solution 1)."""
    R = np.asanyarray(R, dtype=np.float64)
    if np.abs(R[2, 0]) >= 1.0 - 1e-12:
        y = 0.0
        if R[2, 0] < 0:
            p = np.pi / 2
            r = np.arctan2(R[0, 1], R[0, 2])
        else:
            p = -np.pi / 2
            r = np.arctan2(-R[0, 1], -R[0, 2])
    else:
        p = -np.arcsin(R[2, 0])
        r = np.arctan2(R[2, 1] / np.cos(p), R[2, 2] / np.cos(p))
        y = np.arctan2(R[1, 0] / np.cos(p), R[0, 0] / np.cos(p))
    return np.array([r, p, y], dtype=np.float64)


def depth_first_joints(robot, base_link):
    """Joints in depth-first order from the base link, branches in the order the URDF lists them."""
    branches = {}
    for joint in robot.joints:
        branches.setdefault(joint.parent, []).append(joint)
    order = []

    def expand(link):
        for joint in branches.get(link, []):
            order.append(joint)
            expand(joint.child)

    expand(base_link)
    return order


def parse_urdf_annotated(filename, device, xform=None, verbose=False):
    """Kinematic tree of a fixed-base URDF with revolute and fixed joints.

    Returns (UrdfInfo, model). model holds warp arrays named as on warp.sim's finalized Model: joint_type,
    joint_parent and joint_X_p for every joint (the root's fixed joint first), joint_axis for the non-fixed
    joints only, and joint_q with one entry per cspace dim.
    """
    if verbose:
        print("yourdfpy loading:", filename)
    urdf = yourdfpy.URDF.load(filename, load_meshes=False, build_scene_graph=False, build_collision_scene_graph=False)
    xform = wp.transform_identity() if xform is None else xform
    children = {joint.child for joint in urdf.robot.joints}
    base_link = next(link.name for link in urdf.robot.links if link.name not in children)  # the link no joint moves

    cspace_names, cspace2link, link2cspace, cspace_joint_limits = [], [], [-1], []
    link_index = OrderedDict({base_link: 0})
    joint_type, joint_parent, joint_X_p, joint_axis = [int(JOINT_FIXED)], [-1], [xform], []

    for joint in depth_first_joints(urdf.robot, base_link):
        if joint.parent not in link_index:
            raise RuntimeError("Parent not found: {}".format(joint.parent))
        origin = np.eye(4) if joint.origin is None else joint.origin
        rot = wp.quat_rpy(*matrix_to_rpy(origin[:3, :3]))
        joint_X_p.append(wp.transform(origin[:3, 3], rot))
        joint_parent.append(link_index[joint.parent])
        link = len(joint_type)

        if joint.type in ("revolute", "continuous"):
            lower, upper = -1.0e3, 1.0e3
            if joint.limit is not None:
                lower = joint.limit.lower if joint.limit.lower is not None else lower
                upper = joint.limit.upper if joint.limit.upper is not None else upper
            joint_type.append(int(JOINT_REVOLUTE))
            joint_axis.append(wp.normalize(wp.vec3(*joint.axis)))
            cspace_names.append(joint.name)
            cspace2link.append(link)
            cspace_joint_limits.append((lower, upper))
        elif joint.type == "fixed":
            joint_type.append(int(JOINT_FIXED))
        else:
            raise RuntimeError("Joint type not currently supported.")

        link2cspace.append(len(cspace_names) - 1)
        link_index[joint.child] = link

    model = SimpleNamespace(
        device=device,
        joint_type=wp.array(joint_type, dtype=int, device=device),
        joint_parent=wp.array(joint_parent, dtype=int, device=device),
        joint_X_p=wp.array(joint_X_p, dtype=wp.transform, device=device),
        joint_axis=wp.array(joint_axis, dtype=wp.vec3, device=device),
        joint_q=wp.zeros(len(cspace_names), dtype=float, device=device),
        body_count=len(joint_type))
    info = UrdfInfo(cspace_names, link_index, cspace2link, link2cspace, cspace_joint_limits)
    return info, model
