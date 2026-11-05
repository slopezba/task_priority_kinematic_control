from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    config_file = LaunchConfiguration("config_file")
    world_frame = LaunchConfiguration("world_frame")
    task_topic_prefix = LaunchConfiguration("task_topic_prefix")
    interactive_marker_namespace = LaunchConfiguration("interactive_marker_namespace")
    interactive_marker_suffix = LaunchConfiguration("interactive_marker_suffix")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "config_file",
                default_value=PathJoinSubstitution(
                    [
                        FindPackageShare("task_priority_kinematic_control"),
                        "config",
                        "task_priority_target_markers.yaml",
                    ]
                ),
            ),
            DeclareLaunchArgument("world_frame", default_value="world_ned"),
            DeclareLaunchArgument(
                "task_topic_prefix",
                default_value="/cirtesub/controller/task_priority/tasks",
            ),
            DeclareLaunchArgument(
                "interactive_marker_namespace",
                default_value="",
            ),
            DeclareLaunchArgument("interactive_marker_suffix", default_value="interactive_marker"),
            Node(
                package="task_priority_kinematic_control",
                executable="task_priority_target_markers",
                name="task_priority_target_markers",
                output="screen",
                parameters=[
                    config_file,
                    {
                        "world_frame": world_frame,
                        "task_topic_prefix": task_topic_prefix,
                        "interactive_marker_namespace": interactive_marker_namespace,
                        "interactive_marker_suffix": interactive_marker_suffix,
                    }
                ],
            ),
        ]
    )
