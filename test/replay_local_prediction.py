#!/usr/bin/env python3
"""Replay measured Stonefish offsets against KDL/MoveIt and record actual ROS timer intervals."""
import argparse
import copy
import csv
import json
import os
from pathlib import Path
import signal
import subprocess
import tempfile
import time
import xml.etree.ElementTree as ET

import numpy as np
import rclpy
from ament_index_python.packages import get_package_share_directory
from geometry_msgs.msg import PoseStamped, Transform, TwistStamped
from moveit_msgs.msg import AllowedCollisionEntry
from rclpy.action import ActionClient
from sensor_msgs.msg import JointState
from sura_manipulation_actions.action import CheckCartesianTrajectory as Check
from sura_msgs.msg import Navigator
from task_priority_kinematic_control.msg import TaskState
from trajectory_msgs.msg import JointTrajectoryPoint, MultiDOFJointTrajectoryPoint
import yaml


def summary(values):
    if not values:
        return None
    return dict(zip(('p50', 'p95', 'max'), np.percentile(values, [50, 95, 100]).tolist()))


def run(output, directory, posture):
    root = Path(__file__).resolve().parent
    data = json.loads((root / 'data/stonefish_zip_tie_sample.json').read_text())
    description = Path(get_package_share_directory('cirtesub_description'))
    urdf = subprocess.check_output(['xacro', str(description/'urdf/cirtesub_dual_alpha.urdf.xacro')], text=True)
    srdf = subprocess.check_output(['xacro', str(description/'moveit2/cirtesub_dual_alpha.srdf.xacro'),
                                    'floating_base:=true'], text=True)
    config = yaml.safe_load((root.parent/'config/task_priority_kinematic_control.yaml').read_text())
    params = config['task_priority_runtime']['ros__parameters']
    ns = '/local_prediction_replay_' + str(os.getpid())
    params.update(robot_description=urdf, navigator_topic=ns+'/navigation', joint_states_topic=ns+'/joints',
                  base_command_topic=ns+'/tick', left_arm_command_topic=ns+'/left_output',
                  right_arm_command_topic=ns+'/right_output')
    params['tasks.right_pose.use_offset'] = True
    params['robot_description_source_node'] = ''
    config_file = Path(directory)/'runtime.yaml'
    config_file.write_text(yaml.safe_dump({'/**': {'ros__parameters': params}}))
    goal = Check.Goal()
    goal.task_id = 'right_pose'
    goal.tip_frame = data['frame']
    goal.trajectory.header.frame_id = params['world_frame']
    goal.trajectory.duration = 2.0
    goal.tolerance_orientation.w = 1.0
    goal.path_tolerance = [0.02]*3 + [0.3]*3
    goal.goal_tolerance = list(goal.path_tolerance)
    prediction = goal.prediction
    prediction.robot_description = urdf
    prediction.robot_description_semantic = srdf
    prediction.scene.robot_model_name = ET.fromstring(urdf).attrib['name']
    pairs = {(p.attrib['link1'], p.attrib['link2']) for p in ET.fromstring(srdf).findall('disable_collisions')}
    acm = prediction.scene.allowed_collision_matrix
    acm.entry_names = sorted({link for pair in pairs for link in pair})
    acm.entry_values = [AllowedCollisionEntry(enabled=[(a,b) in pairs or (b,a) in pairs
                                                       for b in acm.entry_names]) for a in acm.entry_names]
    joints = JointState()
    names = params['left_arm_joints'] + params['right_arm_joints']
    model_joints = {joint.attrib['name'] for joint in ET.fromstring(urdf).findall('joint')}
    joints.name = [name for name in data['stonefish_joints'] if name in model_joints]
    joints.position = [float(data['stonefish_joints'][name]) for name in joints.name]
    # Match alpha_hardware's signed two-finger reduction to its single gripper coordinate.
    for side, finger in [('left','axis_a1L'), ('right','axis_a1R')]:
        prefix = 'cirtesub/alpha_' + side + '/'
        index = joints.name.index(prefix+'axis_a')
        joints.position[index] = (joints.position[index]-data['stonefish_joints'][prefix+finger]) / (2*51.0)
    if posture == 'nominal':
        for side in ('left','right'):
            ids = params[side+'_arm_joints']
            for name, value in zip(ids,params['tasks.'+side+'_arm_nominal.target']):
                joints.position[joints.name.index(name)] = value
    prediction.scene.robot_state.joint_state = joints
    prediction.global_progress = [0., 1.]
    prediction.horizon = 0.2
    prediction.budget = 3.0
    goal.nominal_joints.joint_names = names
    goal.nominal_base.header.frame_id = params['world_frame']
    goal.nominal_base.joint_names = ['floating_base']
    for i in (0, 1):
        joint = JointTrajectoryPoint(positions=[joints.position[joints.name.index(n)] for n in names])
        joint.time_from_start.sec = 2*i
        base = MultiDOFJointTrajectoryPoint()
        pose = Transform()
        pose.rotation.w = 1.
        base.transforms = [pose]
        base.time_from_start = joint.time_from_start
        goal.nominal_base.points.append(base)
        goal.nominal_joints.points.append(joint)

    node = rclpy.create_node('replay', namespace=ns)
    state = {}
    ticks = []
    node.create_subscription(TaskState, ns+'/task_priority/tasks/right_pose/state',
                             lambda msg: state.update(tip=msg.frame_pose) if msg.active and msg.has_frame_pose else None, 10)
    node.create_subscription(TwistStamped, ns+'/tick',
                             lambda m: ticks.append(m.header.stamp.sec + m.header.stamp.nanosec*1e-9), 100)
    client = ActionClient(node, Check, ns+'/task_priority/check_cartesian_trajectory')
    jpub = node.create_publisher(JointState, ns+'/joints', 10)
    npub = node.create_publisher(Navigator, ns+'/navigation', 10)
    opub = node.create_publisher(PoseStamped, ns+'/task_priority/tasks/right_pose/offset', 10)
    offset = PoseStamped()
    offset.header.frame_id = data['frame']

    def select_offset(i):
        sample = data['samples'][i]
        offset.pose.position.x, offset.pose.position.y, offset.pose.position.z = sample['xyz']
        q = offset.pose.orientation
        q.x,q.y,q.z,q.w = sample['xyzw']

    def publish():
        joints.header.stamp = offset.header.stamp = node.get_clock().now().to_msg()
        jpub.publish(joints)
        npub.publish(Navigator())
        opub.publish(offset)

    select_offset(0)
    node.create_timer(.02, publish)
    def spin_until(condition, timeout=10.0):
        end = time.monotonic()+timeout
        while not condition():
            if process.poll() is not None:
                raise RuntimeError('Task Priority runtime exited. Inspect runtime.log')
            if time.monotonic() > end:
                raise TimeoutError('Timed out waiting for runtime or action')
            rclpy.spin_once(node, timeout_sec=.02)

    with (output/'runtime.log').open('w') as log:
        process = subprocess.Popen(['ros2','run','task_priority_kinematic_control','task_priority_runtime',
                                    '--ros-args','-r','__ns:='+ns,'--params-file',str(config_file)],
                                   stdout=log, stderr=log, start_new_session=True)
        try:
            spin_until(lambda: 'tip' in state and client.server_is_ready(), 20)
            spin_until(lambda: len(ticks) >= 25)
            idle_period = summary((1000*np.diff(ticks)).tolist())
            windows, compute = [], []
            with (output/'trials.csv').open('w') as trials, (output/'steps.csv').open('w') as steps:
                tw, sw = csv.writer(trials), csv.writer(steps)
                tw.writerow(['offset','moving','explore','outcome','feasible','recovered_at_horizon',
                             'configuration_error','candidate_count','planner_compute_ms','request_ms','reason'])
                sw.writerow(['offset','moving','explore','tau','s','confirmed_s','configuration_error','rank',
                             'clearance','active_constraints','ex','ey','ez','rx','ry','rz'])
                for moving in (False, True):
                    for index in range(len(data['samples'])):
                        select_offset(index)
                        count = len(ticks)
                        spin_until(lambda: len(ticks) >= count+3)
                        goal.trajectory.start_pose = copy.deepcopy(state['tip'])
                        goal.trajectory.target_pose = copy.deepcopy(state['tip'])
                        joint_base_end = goal.nominal_base.points[-1].transforms[0]
                        joint_base_end.translation.z = 0.015 if moving else 0.
                        goal.trajectory.target_pose.position.z += joint_base_end.translation.z
                        prediction.start_progress = prediction.maximum_progress = 0. if moving else .4
                        prediction.progress_rate_scale = float(moving)
                        for explore in (False, True):
                            prediction.explore = explore
                            started = time.monotonic()
                            request_start = time.time()
                            sent = client.send_goal_async(goal)
                            spin_until(sent.done)
                            handle = sent.result()
                            if not handle.accepted:
                                raise RuntimeError('Prediction action rejected')
                            future = handle.get_result_async()
                            spin_until(future.done, 15)
                            result = future.result().result
                            windows.append((request_start,time.time()))
                            compute.append(1000*result.compute_seconds)
                            feasible = result.outcome == result.SUCCESS
                            recovered = feasible and result.configuration_error <= prediction.recovery_threshold
                            tw.writerow([index,moving,explore,result.code,feasible,recovered,
                                         result.configuration_error,result.candidates,compute[-1],
                                         1000*(time.monotonic()-started),result.message])
                            for step in result.trace:
                                sw.writerow([index,moving,explore,step.elapsed,step.progress,step.confirmed_progress,
                                             step.configuration_error,step.hierarchy_rank,step.clearance,
                                             ' '.join(step.active_constraints),*step.cartesian_error])
            count = len(ticks)
            spin_until(lambda: len(ticks) >= count + 2)
            periods = [1000*(b-a) for a,b in zip(ticks,ticks[1:])
                       if any(start <= b and stop >= a for start,stop in windows)]
            stats = {'idle_runtime_period_ms': idle_period,'prediction_runtime_period_ms': summary(periods),
                     'prediction_compute_ms': summary(compute), 'requests':len(compute)}
            (output/'timing.json').write_text(json.dumps(stats,indent=2)+'\n')
            print(json.dumps(stats))
        finally:
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGINT)
                try:
                    process.wait(timeout=8)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=8)
            node.destroy_node()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--posture', choices=['captured','nominal'], default='captured')
    parser.add_argument('--domain', type=int, default=186)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ['ROS_DOMAIN_ID'] = str(args.domain)
    os.environ['ROS_LOCALHOST_ONLY'] = '1'
    rclpy.init()
    try:
        with tempfile.TemporaryDirectory() as directory:
            run(args.output,directory,args.posture)
    finally:
        rclpy.shutdown()
