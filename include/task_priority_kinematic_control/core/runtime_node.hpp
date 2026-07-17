#pragma once

#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"
#include "task_priority_kinematic_control/core/task_manager.hpp"
#include "task_priority_kinematic_control/core/whole_body_model.hpp"
#include "task_priority_kinematic_control/kinematics/kinematics_backend.hpp"

#include "task_priority_kinematic_control/msg/hierarchy_state.hpp"
#include "task_priority_kinematic_control/msg/solver_diagnostics.hpp"
#include "task_priority_kinematic_control/srv/list_tasks.hpp"
#include "task_priority_kinematic_control/srv/reorder_tasks.hpp"
#include "task_priority_kinematic_control/srv/set_task_enabled.hpp"
#include "task_priority_kinematic_control/srv/switch_backend.hpp"

#include <control_msgs/action/follow_joint_trajectory.hpp>
#include <geometry_msgs/msg/pose_stamped.hpp>
#include <geometry_msgs/msg/twist_stamped.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sensor_msgs/msg/joint_state.hpp>
#include <std_msgs/msg/float64_multi_array.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <sura_msgs/msg/navigator.hpp>
#include <trajectory_msgs/msg/joint_trajectory.hpp>

#include <pluginlib/class_loader.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp/parameter_client.hpp>

#include <map>

namespace task_priority_kinematic_control
{

class RuntimeNode : public rclcpp::Node
{
public:
  explicit RuntimeNode(const rclcpp::NodeOptions & options = rclcpp::NodeOptions());

private:
  void configure_model();
  void configure_backend();
  void configure_solver();
  void configure_publishers();
  void configure_subscribers();
  void configure_services();
  void configure_timer();
  void declare_parameters();
  std::string resolve_robot_description();
  void rebuild_backend();

  void navigator_callback(const sura_msgs::msg::Navigator::SharedPtr msg);
  void joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg);
  void timer_callback();
  void publish_status();
  void execute_task_trajectory_goal(
    const std::string & task_id,
    const std::shared_ptr<rclcpp_action::ServerGoalHandle<
      control_msgs::action::FollowJointTrajectory>> goal_handle);
  rcl_interfaces::msg::SetParametersResult on_parameters_set(
    const std::vector<rclcpp::Parameter> & parameters);

  std::shared_ptr<WholeBodyModel> model_;
  WholeBodyState state_;
  std::unique_ptr<TaskManager> task_manager_;
  HierarchySolver solver_;

  pluginlib::ClassLoader<KinematicsBackend> backend_loader_;
  KinematicsBackendPtr backend_;
  std::string backend_plugin_name_;
  std::shared_ptr<rclcpp::SyncParametersClient> remote_param_client_;
  std::map<std::string, rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr>
    task_target_subs_;
  std::map<std::string, rclcpp::Subscription<std_msgs::msg::Float64MultiArray>::SharedPtr>
    task_joint_target_subs_;
  std::map<std::string, rclcpp::Subscription<trajectory_msgs::msg::JointTrajectory>::SharedPtr>
    task_joint_trajectory_subs_;
  std::map<std::string, rclcpp_action::Server<control_msgs::action::FollowJointTrajectory>::SharedPtr>
    task_joint_trajectory_action_servers_;

  rclcpp::Subscription<sura_msgs::msg::Navigator>::SharedPtr navigator_sub_;
  rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr joint_state_sub_;

  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr base_command_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr left_arm_command_pub_;
  rclcpp::Publisher<std_msgs::msg::Float64MultiArray>::SharedPtr right_arm_command_pub_;
  rclcpp::Publisher<msg::HierarchyState>::SharedPtr hierarchy_pub_;
  rclcpp::Publisher<msg::SolverDiagnostics>::SharedPtr diagnostics_pub_;

  rclcpp::Service<srv::SetTaskEnabled>::SharedPtr set_task_enabled_srv_;
  rclcpp::Service<srv::ReorderTasks>::SharedPtr reorder_tasks_srv_;
  rclcpp::Service<srv::ListTasks>::SharedPtr list_tasks_srv_;
  rclcpp::Service<srv::SwitchBackend>::SharedPtr switch_backend_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;

  rclcpp::TimerBase::SharedPtr timer_;
  OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
};

}  // namespace task_priority_kinematic_control
