from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import LaunchConfiguration
from launch.actions import DeclareLaunchArgument


def generate_launch_description():
    params_file = LaunchConfiguration("params_file")

    return LaunchDescription(
        [
            DeclareLaunchArgument(
                "params_file",
                default_value="",
                description="Path to the task priority controller parameter file.",
            ),
            Node(
                package="task_priority_kinematic_control",
                executable="task_priority_runtime",
                name="task_priority_runtime",
                output="screen",
                parameters=[params_file] if str(params_file) else [],
            ),
        ]
    )
