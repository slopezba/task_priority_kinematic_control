#pragma once

#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"
#include "task_priority_kinematic_control/core/task_manager.hpp"
#include "task_priority_kinematic_control/core/whole_body_model.hpp"
#include "task_priority_kinematic_control/kinematics/kinematics_backend.hpp"
#include "task_priority_kinematic_control/msg/controller_output.hpp"
#include "task_priority_kinematic_control/msg/hierarchy_state.hpp"
#include "task_priority_kinematic_control/msg/task_state.hpp"
#include "task_priority_kinematic_control/srv/list_tasks.hpp"
#include "task_priority_kinematic_control/srv/reorder_tasks.hpp"
#include "task_priority_kinematic_control/srv/set_task_disabled.hpp"
#include "task_priority_kinematic_control/srv/set_task_enabled.hpp"

#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "sura_msgs/msg/navigator.hpp"

#include <mutex>
#include <pluginlib/class_loader.hpp>
#include <rclcpp/parameter_client.hpp>

#include <map>
#include <memory>
#include <string>
#include <vector>

namespace task_priority_kinematic_control
{

class TaskPriorityController : public controller_interface::ControllerInterface
{
public:
  controller_interface::CallbackReturn on_init() override;

  controller_interface::InterfaceConfiguration command_interface_configuration() const override;
  controller_interface::InterfaceConfiguration state_interface_configuration() const override;

  controller_interface::CallbackReturn on_configure(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_activate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_deactivate(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_cleanup(
    const rclcpp_lifecycle::State & previous_state) override;

  controller_interface::CallbackReturn on_shutdown(
    const rclcpp_lifecycle::State & previous_state) override;

protected:
  controller_interface::return_type update(
    const rclcpp::Time & time,
    const rclcpp::Duration & period) override;

private:
  using NavigatorMsg = sura_msgs::msg::Navigator;
  using PoseStamped = geometry_msgs::msg::PoseStamped;
  using Float64MultiArray = std_msgs::msg::Float64MultiArray;

  void declare_task_parameters();
  std::string resolve_robot_description();
  void rebuild_backend();
  void configure_model();
  void configure_solver();
  void configure_external_interfaces();
  void publish_hierarchy_state(const rclcpp::Time & time);
  void publish_task_states(const std::vector<TaskComputation> & task_outputs);
  void publish_controller_output(const rclcpp::Time & time, const WholeBodyCommand & command);
  void refresh_task_manager();
  void reset_commands();
  void publish_zero_controller_output(const rclcpp::Time & time);
  rcl_interfaces::msg::SetParametersResult on_parameters_set(
    const std::vector<rclcpp::Parameter> & parameters);

  std::shared_ptr<WholeBodyModel> model_;
  WholeBodyState state_;
  std::unique_ptr<TaskManager> task_manager_;
  HierarchySolver solver_;

  std::unique_ptr<pluginlib::ClassLoader<KinematicsBackend>> backend_loader_;
  KinematicsBackendPtr backend_;
  std::shared_ptr<rclcpp::SyncParametersClient> remote_param_client_;

  rclcpp::Subscription<NavigatorMsg>::SharedPtr navigator_sub_;
  std::map<std::string, rclcpp::Subscription<PoseStamped>::SharedPtr> task_target_subs_;
  std::map<std::string, rclcpp::Subscription<Float64MultiArray>::SharedPtr> task_joint_target_subs_;
  rclcpp::Publisher<msg::HierarchyState>::SharedPtr hierarchy_state_pub_;
  rclcpp::Publisher<msg::ControllerOutput>::SharedPtr controller_output_pub_;
  std::map<std::string, rclcpp::Publisher<msg::TaskState>::SharedPtr> task_state_pubs_;
  rclcpp::Service<srv::ListTasks>::SharedPtr list_tasks_srv_;
  rclcpp::Service<srv::SetTaskEnabled>::SharedPtr set_task_enabled_srv_;
  rclcpp::Service<srv::SetTaskDisabled>::SharedPtr set_task_disabled_srv_;
  rclcpp::Service<srv::ReorderTasks>::SharedPtr reorder_tasks_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_srv_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
  realtime_tools::RealtimeBuffer<std::shared_ptr<NavigatorMsg>> navigator_buffer_;

  std::string backend_plugin_name_;
  std::string body_velocity_controller_name_;
  std::vector<std::string> left_arm_joints_;
  std::vector<std::string> right_arm_joints_;
  std::vector<std::string> task_ids_;
  std::vector<int> active_base_dofs_;
  std::mutex task_mutex_;
};

}  // namespace task_priority_kinematic_control
