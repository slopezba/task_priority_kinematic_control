#pragma once

#include "task_priority_kinematic_control/core/pose_task_io.hpp"
#include "task_priority_kinematic_control/core/cartesian_execution_io.hpp"
#include "task_priority_kinematic_control/core/manual_task_io.hpp"
#include "task_priority_kinematic_control/core/navigation_sample.hpp"

#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"
#include "task_priority_kinematic_control/core/task_manager.hpp"
#include "task_priority_kinematic_control/core/capsule_observer.hpp"
#include "task_priority_kinematic_control/core/whole_body_model.hpp"
#include "task_priority_kinematic_control/kinematics/kinematics_backend.hpp"
#include "task_priority_kinematic_control/msg/controller_output.hpp"
#include "task_priority_kinematic_control/msg/hierarchy_state.hpp"
#include "task_priority_kinematic_control/msg/task_state.hpp"
#include "task_priority_kinematic_control/srv/list_tasks.hpp"
#include "task_priority_kinematic_control/srv/reorder_tasks.hpp"
#include "task_priority_kinematic_control/srv/set_solver_config.hpp"
#include "task_priority_kinematic_control/srv/set_task_disabled.hpp"
#include "task_priority_kinematic_control/srv/set_task_enabled.hpp"
#include "task_priority_kinematic_control/srv/set_task_gains.hpp"
#include "task_priority_kinematic_control/srv/set_task_joint_activation.hpp"

#include "control_msgs/action/follow_joint_trajectory.hpp"
#include "controller_interface/controller_interface.hpp"
#include "geometry_msgs/msg/pose_stamped.hpp"
#include "geometry_msgs/msg/twist.hpp"
#include "realtime_tools/realtime_publisher.hpp"
#include "rcl_interfaces/msg/set_parameters_result.hpp"
#include "rclcpp_action/rclcpp_action.hpp"
#include "rclcpp/rclcpp.hpp"
#include "realtime_tools/realtime_buffer.hpp"
#include "std_msgs/msg/float64_multi_array.hpp"
#include "std_msgs/msg/bool.hpp"
#include "std_srvs/srv/trigger.hpp"
#include "sura_msgs/msg/navigator.hpp"
#include "trajectory_msgs/msg/joint_trajectory.hpp"

#include <atomic>
#include <condition_variable>
#include <deque>
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
  friend class TrajectoryActionReadinessTest;

  using NavigatorMsg = sura_msgs::msg::Navigator;
  using PoseStamped = geometry_msgs::msg::PoseStamped;
  using Float64MultiArray = std_msgs::msg::Float64MultiArray;
  using JointTrajectory = trajectory_msgs::msg::JointTrajectory;
  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
  using FollowJointTrajectoryGoalHandle = rclcpp_action::ServerGoalHandle<FollowJointTrajectory>;

  struct RuntimeGainUpdate
  {
    std::string task_id;
    std::string field;
    std::vector<double> values;
  };

  struct RuntimeTuningCommand
  {
    enum class Type
    {
      Solver,
      Gains
    };

    Type type = Type::Solver;
    std::string solver_method;
    double dls_lambda = 0.0;
    bool update_dof_weights = false;
    std::vector<double> dof_weights;
    std::vector<RuntimeGainUpdate> gain_updates;
    bool completed = false;
    bool canceled = false;
    bool success = false;
    std::string message;
    std::mutex mutex;
    std::condition_variable cv;
  };

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
  bool apply_runtime_tuning_command(RuntimeTuningCommand & command, std::string & message);
  void process_pending_runtime_tuning();
  bool enqueue_runtime_tuning_command(
    const std::shared_ptr<RuntimeTuningCommand> & command,
    std::string & message);
  void reset_commands();
  void publish_base_command(const Eigen::Matrix<double, 6, 1> & command);
  void publish_zero_controller_output(const rclcpp::Time & time);
  void execute_task_trajectory_goal(
    const std::string & task_id,
    const std::shared_ptr<FollowJointTrajectoryGoalHandle> goal_handle);
  void poll_task_trajectory_goals();
  void stop_task_trajectory_goals(const std::string & reason);
  bool trajectory_task_enabled_locked(const std::string & task_id) const;

  struct TrajectoryActionSlot
  {
    rclcpp_action::GoalUUID uuid;
    std::shared_ptr<FollowJointTrajectoryGoalHandle> handle;
  };
  rcl_interfaces::msg::SetParametersResult on_parameters_set(
    const std::vector<rclcpp::Parameter> & parameters);

  std::mutex task_mutex_;
  std::shared_ptr<WholeBodyModel> model_;
  WholeBodyState state_;
  std::unique_ptr<TaskManager> task_manager_;
  HierarchySolver solver_;
  PoseTaskIO pose_task_io_;
  ManualTaskIO manual_task_io_;
  CartesianTrajectoryCoordinator cartesian_coordinator_;
  CartesianExecutionIO cartesian_execution_io_;
  std::unique_ptr<CapsuleObserver> capsule_observer_;

  std::unique_ptr<pluginlib::ClassLoader<KinematicsBackend>> backend_loader_;
  KinematicsBackendPtr backend_;
  std::shared_ptr<rclcpp::SyncParametersClient> remote_param_client_;

  rclcpp::Subscription<NavigatorMsg>::SharedPtr navigator_sub_;
  std::map<std::string, rclcpp::Subscription<PoseStamped>::SharedPtr> task_target_subs_;
  std::map<std::string, rclcpp::Subscription<Float64MultiArray>::SharedPtr> task_joint_target_subs_;
  std::map<std::string, rclcpp::Subscription<JointTrajectory>::SharedPtr> task_joint_trajectory_subs_;
  std::map<std::string, rclcpp_action::Server<FollowJointTrajectory>::SharedPtr>
    task_joint_trajectory_action_servers_;
  rclcpp::Publisher<msg::HierarchyState>::SharedPtr hierarchy_state_pub_;
  rclcpp::Publisher<msg::ControllerOutput>::SharedPtr controller_output_pub_;
  std::map<std::string, rclcpp::Publisher<msg::TaskState>::SharedPtr> task_state_pubs_;
  rclcpp::Service<srv::ListTasks>::SharedPtr list_tasks_srv_;
  rclcpp::Service<srv::SetTaskEnabled>::SharedPtr set_task_enabled_srv_;
  rclcpp::Service<srv::SetTaskDisabled>::SharedPtr set_task_disabled_srv_;
  rclcpp::Service<srv::SetSolverConfig>::SharedPtr set_solver_config_srv_;
  rclcpp::Service<srv::SetTaskGains>::SharedPtr set_task_gains_srv_;
  rclcpp::Service<srv::SetTaskJointActivation>::SharedPtr set_task_joint_activation_srv_;
  rclcpp::Service<srv::ReorderTasks>::SharedPtr reorder_tasks_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr stop_srv_;
  rclcpp::node_interfaces::OnSetParametersCallbackHandle::SharedPtr param_callback_handle_;
  realtime_tools::RealtimeBuffer<NavigationSample> navigator_buffer_;
  // Accessed under task_mutex_; only update() consumes the realtime buffer.
  NavigationSample navigation_sample_;

  std::string backend_plugin_name_;
  std::string body_velocity_controller_name_;
  rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr active_pub_;
  bool base_command_via_topic_{false};
  double navigator_timeout_{0.0};
  rclcpp::Publisher<geometry_msgs::msg::Twist>::SharedPtr base_command_pub_;
  std::shared_ptr<realtime_tools::RealtimePublisher<geometry_msgs::msg::Twist>> base_command_rt_pub_;
  std::vector<std::string> left_arm_joints_;
  std::vector<std::string> right_arm_joints_;
  std::vector<std::string> task_ids_;
  std::vector<int> active_base_dofs_;
  std::mutex runtime_tuning_mutex_;
  std::deque<std::shared_ptr<RuntimeTuningCommand>> pending_runtime_tuning_;
  std::atomic_bool parameter_update_pending_{false};
  bool trajectory_execution_ready_{false};
  std::map<std::string, TrajectoryActionSlot> trajectory_action_slots_;
  rclcpp::TimerBase::SharedPtr trajectory_result_timer_;
};

}  // namespace task_priority_kinematic_control
