#include "task_priority_kinematic_control/core/runtime_node.hpp"

#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <algorithm>
#include <chrono>
#include <functional>
#include <stdexcept>
#include <thread>

namespace task_priority_kinematic_control
{

namespace
{

template<typename T>
void declare_if_missing(rclcpp::Node & node, const std::string & name, const T & default_value)
{
  if (!node.has_parameter(name)) {
    node.declare_parameter<T>(name, default_value);
  }
}

SolverMethod parse_solver_method(const std::string & name)
{
  if (name == "pinv") {
    return SolverMethod::kPinv;
  }
  if (name == "svd") {
    return SolverMethod::kSvd;
  }
  return SolverMethod::kDls;
}

std::string solver_method_name(SolverMethod method)
{
  switch (method) {
    case SolverMethod::kPinv:
      return "pinv";
    case SolverMethod::kSvd:
      return "svd";
    case SolverMethod::kDls:
    default:
      return "dls";
  }
}

std::vector<int> parse_active_base_dofs(
  const std::vector<int64_t> & param_values,
  const rclcpp::Logger & logger)
{
  std::ostringstream values_stream;
  values_stream << "[";

  std::vector<int> parsed_values;
  parsed_values.reserve(param_values.size());
  for (size_t i = 0; i < param_values.size(); ++i) {
    const auto value = param_values[i];
    if (i > 0) {
      values_stream << ", ";
    }
    values_stream << value;

    if (value < 0 || value > 5) {
      throw std::runtime_error(
        "Invalid active_base_dofs parameter. Expected values in [0, 5], got " +
        std::to_string(value) + " at index " + std::to_string(i));
    }
    parsed_values.push_back(static_cast<int>(value));
  }

  values_stream << "]";
  RCLCPP_INFO(
    logger,
    "Raw active_base_dofs parameter: %s",
    values_stream.str().c_str());

  return parsed_values;
}

bool split_task_parameter_name(
  const std::string & name,
  std::string & task_id,
  std::string & field)
{
  const std::string prefix = "tasks.";
  if (name.rfind(prefix, 0) != 0) {
    return false;
  }

  const auto field_separator = name.rfind('.');
  if (field_separator == std::string::npos || field_separator <= prefix.size()) {
    return false;
  }

  task_id = name.substr(prefix.size(), field_separator - prefix.size());
  field = name.substr(field_separator + 1);
  return !task_id.empty() && !field.empty();
}

bool values_are_non_negative(const std::vector<double> & values)
{
  return std::all_of(values.begin(), values.end(), [](double value) {
    return value >= 0.0;
  });
}

}  // namespace

RuntimeNode::RuntimeNode(const rclcpp::NodeOptions & options)
: rclcpp::Node("task_priority_runtime", options),
  model_(std::make_shared<WholeBodyModel>()),
  task_manager_(nullptr),
  backend_loader_("task_priority_kinematic_control", "task_priority_kinematic_control::KinematicsBackend")
{
  task_manager_ = std::make_unique<TaskManager>(
    this->get_node_parameters_interface(), this->get_logger());
  declare_parameters();
  configure_model();
  configure_backend();
  configure_solver();
  task_manager_->configure(TaskContext{model_, backend_, get_clock()});
  capsule_observer_ = std::make_unique<CapsuleObserver>(task_manager_->tasks(),
    get_fully_qualified_name(), get_node_base_interface()->get_context());
  configure_publishers();
  configure_subscribers();
  configure_services();
  configure_timer();
  param_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&RuntimeNode::on_parameters_set, this, std::placeholders::_1));
}

void RuntimeNode::declare_parameters()
{
  CartesianExecutionIO::declare_parameters(this);
  declare_if_missing<std::string>(*this, "backend_plugin", "task_priority_kinematic_control/KDLKinematicsBackend");
  declare_if_missing<std::string>(*this, "robot_description", "");
  declare_if_missing<std::string>(*this, "robot_description_source_node", "/robot_state_publisher");
  declare_if_missing<double>(*this, "robot_description_wait_timeout_sec", 3.0);
  declare_if_missing<std::string>(*this, "world_frame", "world_ned");
  declare_if_missing<std::string>(*this, "base_frame", "cirtesub/base_link");
  declare_if_missing<std::string>(*this, "left_tip_frame", "cirtesub/alpha_left/ee_base_link");
  declare_if_missing<std::string>(*this, "right_tip_frame", "cirtesub/alpha_right/ee_base_link");
  declare_if_missing<std::vector<std::string>>(*this, "left_arm_joints", {});
  declare_if_missing<std::vector<std::string>>(*this, "right_arm_joints", {});
  declare_if_missing<std::vector<int64_t>>(*this, "active_base_dofs", {0, 1, 2, 3, 4, 5});
  declare_if_missing<std::string>(*this, "solver_method", "dls");
  declare_if_missing<double>(*this, "dls_lambda", 0.05);
  declare_if_missing<std::vector<double>>(*this, "dof_weights", {});
  declare_if_missing<std::vector<double>>(*this, "velocity_limits", {});
  declare_if_missing<double>(*this, "rate_hz", 20.0);
  declare_if_missing<std::string>(*this, "navigator_topic", "/cirtesub/navigator/navigation");
  declare_if_missing<std::string>(*this, "joint_states_topic", "/cirtesub/alpha/joint_states");
  declare_if_missing<std::string>(*this, "base_command_topic", "/cirtesub/controller/task_priority/base_cmd");
  declare_if_missing<std::string>(*this, "left_arm_command_topic", "/cirtesub/alpha/controller/alpha_left_forward_velocity_controller/commands");
  declare_if_missing<std::string>(*this, "right_arm_command_topic", "/cirtesub/alpha/controller/alpha_right_forward_velocity_controller/commands");
  declare_if_missing<std::vector<std::string>>(*this, "task_ids", {});
  declare_if_missing<double>(*this, "navigator_timeout", 0.5);
  declare_if_missing<double>(*this, "joint_state_timeout", 0.5);
  navigator_timeout_ = get_parameter("navigator_timeout").as_double();
  joint_state_timeout_ = get_parameter("joint_state_timeout").as_double();
  for (const auto & task_id : this->get_parameter("task_ids").as_string_array()) {
    const std::string prefix = "tasks." + task_id + ".";
    declare_if_missing<std::string>(*this, prefix + "plugin", "");
    const bool stabilize_limits = this->get_parameter(prefix + "plugin").as_string() ==
      "task_priority_kinematic_control/StabilizeLimitsTask";
    declare_if_missing<bool>(*this, prefix + "enabled", true);
    declare_if_missing<bool>(*this, prefix + "auto_update_target_pose", true);
    declare_if_missing<bool>(*this, prefix + "roll_pitch_control", false);
    declare_if_missing<bool>(*this, prefix + "allow_roll_pitch", false);
    declare_if_missing<double>(*this, prefix + "feedforward_timeout", 0.0);
    declare_if_missing<double>(*this, prefix + "joint_command_timeout", 0.5);
    declare_if_missing<double>(*this, prefix + "feedforward_velocity_threshold", 0.001);
    declare_if_missing<double>(*this, prefix + "priority", 0.0);
    declare_if_missing<std::string>(*this, prefix + "group", "default");
    if (this->get_parameter(prefix + "plugin").as_string() ==
      "task_priority_kinematic_control/SelfCollisionAvoidanceTask") {
      declare_collision_parameters(get_node_parameters_interface(), prefix);
    } else if (this->get_parameter(prefix + "plugin").as_string() ==
      "task_priority_kinematic_control/JointLimitsTask") {
      declare_joint_limits_marker_parameters(get_node_parameters_interface(), prefix);
    } else if (stabilize_limits) {
      declare_stabilize_limits_debug_parameters(get_node_parameters_interface(), prefix);
    }
    declare_if_missing<std::string>(*this, prefix + "frame_id", "");
    declare_if_missing<std::vector<double>>(*this, prefix + "gain", {});
    declare_if_missing<std::vector<double>>(*this, prefix + "lower_limits", {});
    declare_if_missing<std::vector<double>>(*this, prefix + "upper_limits", {});
    declare_if_missing<double>(*this, prefix + "margin", 0.1);
    declare_if_missing<double>(*this, prefix + "alpha", stabilize_limits ? 0.06981317 : 0.1);
    declare_if_missing<double>(*this, prefix + "delta", stabilize_limits ? 0.13962634 : 0.15);
    if (stabilize_limits) {
      declare_if_missing<double>(*this, prefix + "max_repulsive_velocity", 0.3);
    }
    declare_if_missing<double>(*this, prefix + "eps", 1e-4);
    declare_if_missing<double>(*this, prefix + "gain_scalar",
      stabilize_limits ? 2.0 :
      (this->get_parameter(prefix + "plugin").as_string() ==
      "task_priority_kinematic_control/SelfCollisionAvoidanceTask" ? 1.0 : 0.5));
    declare_if_missing<std::vector<std::string>>(*this, prefix + "joint_names", {});
    declare_if_missing<std::vector<double>>(*this, prefix + "target", {});
    declare_if_missing<std::vector<int64_t>>(*this, prefix + "activation", {});
    declare_if_missing<double>(*this, prefix + "target_yaw", 0.0);
    declare_if_missing<std::vector<double>>(*this, prefix + "goal_tolerance", {});
    declare_if_missing<double>(*this, prefix + "trajectory_timeout", 0.5);
    declare_if_missing<bool>(*this, prefix + "hold_last_point", true);
    declare_if_missing<std::string>(*this, prefix + "interpolation_method", "linear");
    declare_if_missing<double>(*this, prefix + "relative_pose_tolerance", 1.0e-4);
    declare_if_missing<double>(*this, prefix + "planning_timeout", 5.0);
    declare_if_missing<double>(*this, prefix + "move_gain", 0.8);
    declare_if_missing<double>(*this, prefix + "move_goal_tolerance", 0.04);
    declare_if_missing<double>(*this, prefix + "blend_duration", 2.0);
    declare_if_missing<double>(*this, prefix + "max_joint_velocity", 0.25);
    declare_if_missing<double>(*this, prefix + "min_sigma", 0.03);
    declare_if_missing<double>(*this, prefix + "min_joint_margin", 0.02);
    declare_if_missing<double>(*this, prefix + "min_collision_margin", 0.04);
    declare_if_missing<double>(*this, prefix + "nullspace_probe_radius", 0.12);
    declare_if_missing<double>(*this, prefix + "nullspace_probe_steps", 5.0);
    declare_if_missing<double>(*this, prefix + "reprojection_iterations", 20.0);
    declare_if_missing<double>(*this, prefix + "reprojection_tolerance", 1.0e-4);
    declare_if_missing<double>(*this, prefix + "preferred_joint_margin", 0.08);
    declare_if_missing<bool>(*this, prefix + "debug", false);
    declare_if_missing<bool>(*this, prefix + "fallback_to_servo_on_plan_failure", true);
    declare_if_missing<bool>(*this, prefix + "global_search.enabled", false);
    declare_if_missing<double>(*this, prefix + "global_search.random_starts", 128.0);
    declare_if_missing<double>(*this, prefix + "global_search.local_perturbation_starts", 12.0);
    declare_if_missing<double>(*this, prefix + "global_search.seed", 7.0);
    declare_if_missing<double>(*this, prefix + "global_search.projection_iterations", 20.0);
    declare_if_missing<double>(*this, prefix + "global_search.projection_tolerance", 1.0e-4);
    declare_if_missing<double>(*this, prefix + "global_search.solution_tolerance", 5.0e-4);
    declare_if_missing<double>(*this, prefix + "global_search.explore_threshold", 1.0e-4);
    declare_if_missing<double>(*this, prefix + "global_search.deduplicate_radius", 0.004);
    declare_if_missing<double>(*this, prefix + "global_search.connect_radius", 1.25);
    declare_if_missing<double>(*this, prefix + "global_search.max_edge_projection_distance", 0.08);
    declare_if_missing<double>(*this, prefix + "global_search.max_edge_neighbors", 2.0);
    declare_if_missing<double>(*this, prefix + "global_search.path_check_steps", 2.0);
    declare_if_missing<double>(*this, prefix + "global_search.max_components_to_plan", 2.0);
    declare_if_missing<double>(*this, prefix + "global_search.polar_angular_samples", 8.0);
    declare_if_missing<double>(*this, prefix + "global_search.polar_radial_step", 0.06);
    declare_if_missing<double>(*this, prefix + "global_search.polar_max_radius", 0.36);
    declare_if_missing<double>(*this, prefix + "global_search.max_charts_per_component", 3.0);
    declare_if_missing<double>(*this, prefix + "global_search.max_projector_change", 0.35);
    declare_if_missing<double>(*this, prefix + "global_search.atlas_max_charts_per_seed", 3.0);
    declare_if_missing<double>(*this, prefix + "global_search.atlas_max_total_charts", 160.0);
    declare_if_missing<double>(*this, prefix + "global_search.atlas_step", 0.08);
    declare_if_missing<double>(*this, prefix + "global_search.atlas_neighbor_radius", 0.12);
    declare_if_missing<bool>(*this, prefix + "global_search.projection_trace.enabled", true);
    declare_if_missing<double>(*this, prefix + "global_search.projection_trace.seed_index", 0.0);
    declare_if_missing<std::string>(
      *this,
      prefix + "global_search.debug_dump_path",
      "/tmp/end_effector_plan_relative_pose_manifold_debug.txt");
    declare_if_missing<bool>(*this, prefix + "external_planner.enabled", false);
    declare_if_missing<std::string>(
      *this,
      prefix + "external_planner.action_name",
      "/cirtesub/manipulation/plan_bimanual_joint_trajectory");
    declare_if_missing<double>(*this, prefix + "external_planner.wait_timeout", 1.0);
    declare_if_missing<double>(*this, prefix + "external_planner.result_timeout", 5.0);
    declare_if_missing<double>(*this, prefix + "external_planner.goal_tolerance", 0.02);
  }
}

void RuntimeNode::configure_model()
{
  const auto active_base_dofs = parse_active_base_dofs(
    this->get_parameter("active_base_dofs").as_integer_array(),
    this->get_logger());

  model_->configure(
    this->get_parameter("world_frame").as_string(),
    this->get_parameter("base_frame").as_string(),
    this->get_parameter("left_tip_frame").as_string(),
    this->get_parameter("right_tip_frame").as_string(),
    this->get_parameter("left_arm_joints").as_string_array(),
    this->get_parameter("right_arm_joints").as_string_array(),
    active_base_dofs);
}

void RuntimeNode::configure_backend()
{
  rebuild_backend();
}

std::string RuntimeNode::resolve_robot_description()
{
  const std::string local_description = this->get_parameter("robot_description").as_string();
  if (!local_description.empty()) {
    return local_description;
  }

  const std::string source_node = this->get_parameter("robot_description_source_node").as_string();
  const double timeout_sec = std::max(
    0.1, this->get_parameter("robot_description_wait_timeout_sec").as_double());

  auto non_owning_node = std::shared_ptr<rclcpp::Node>(this, [](rclcpp::Node *) {});
  remote_param_client_ = std::make_shared<rclcpp::SyncParametersClient>(non_owning_node, source_node);

  const auto timeout = std::chrono::duration<double>(timeout_sec);
  if (!remote_param_client_->wait_for_service(timeout)) {
    throw std::runtime_error(
      "robot_description is empty locally and parameter service for '" + source_node +
      "' is not available");
  }

  const auto parameters = remote_param_client_->get_parameters({"robot_description"});
  if (parameters.empty() || parameters.front().get_type() != rclcpp::ParameterType::PARAMETER_STRING) {
    throw std::runtime_error(
      "Failed to read string parameter 'robot_description' from '" + source_node + "'");
  }

  const std::string remote_description = parameters.front().as_string();
  if (remote_description.empty()) {
    throw std::runtime_error(
      "Node '" + source_node + "' returned an empty robot_description");
  }

  RCLCPP_INFO(
    this->get_logger(),
    "Loaded robot_description at runtime from '%s'",
    source_node.c_str());
  return remote_description;
}

void RuntimeNode::rebuild_backend()
{
  backend_plugin_name_ = this->get_parameter("backend_plugin").as_string();
  backend_ = backend_loader_.createSharedInstance(backend_plugin_name_);
  backend_->configure(*model_, resolve_robot_description(), this->get_logger());
  WholeBodyState neutral;
  neutral.joint_positions = Eigen::VectorXd::Zero(model_->all_joint_names().size());
  neutral.joint_velocities = neutral.joint_positions;
  backend_->update(neutral);
}

void RuntimeNode::configure_solver()
{
  solver_.configure(model_->total_dofs());
  solver_.set_method(parse_solver_method(this->get_parameter("solver_method").as_string()));
  solver_.set_damping(this->get_parameter("dls_lambda").as_double());

  const auto weights = this->get_parameter("dof_weights").as_double_array();
  if (!weights.empty()) {
    Eigen::VectorXd w(weights.size());
    for (size_t i = 0; i < weights.size(); ++i) {
      w(static_cast<Eigen::Index>(i)) = weights[i];
    }
    solver_.set_weights(w);
  }

  const auto limits = this->get_parameter("velocity_limits").as_double_array();
  if (!limits.empty()) {
    Eigen::VectorXd v(limits.size());
    for (size_t i = 0; i < limits.size(); ++i) {
      v(static_cast<Eigen::Index>(i)) = limits[i];
    }
    solver_.set_velocity_limits(v);
  }
}

void RuntimeNode::configure_publishers()
{
  for (const auto & task : task_manager_->tasks()) {
    task_state_pubs_[task->id()] = create_publisher<msg::TaskState>(
      "task_priority/tasks/" + task->id() + "/state", rclcpp::SystemDefaultsQoS());
  }
  base_command_pub_ = this->create_publisher<geometry_msgs::msg::TwistStamped>(
    this->get_parameter("base_command_topic").as_string(), 10);
  left_arm_command_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
    this->get_parameter("left_arm_command_topic").as_string(), 10);
  right_arm_command_pub_ = this->create_publisher<std_msgs::msg::Float64MultiArray>(
    this->get_parameter("right_arm_command_topic").as_string(), 10);
  hierarchy_pub_ = this->create_publisher<msg::HierarchyState>("hierarchy_state", 10);
  diagnostics_pub_ = this->create_publisher<msg::SolverDiagnostics>("solver_diagnostics", 10);
}

void RuntimeNode::configure_subscribers()
{
  pose_task_io_.configure(this, task_manager_->tasks(), model_->world_frame(), model_->base_frame(),
    &task_mutex_, [this](const std::string & task, std::string & message) {return task_manager_->allow_mutation(task, message);});
  pose_task_io_.set_active(true);
  navigator_sub_ = this->create_subscription<sura_msgs::msg::Navigator>(
    this->get_parameter("navigator_topic").as_string(),
    10,
    std::bind(&RuntimeNode::navigator_callback, this, std::placeholders::_1));
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    this->get_parameter("joint_states_topic").as_string(),
    10,
    std::bind(&RuntimeNode::joint_state_callback, this, std::placeholders::_1));

  task_target_subs_.clear();
  cartesian_execution_io_.configure(this, cartesian_coordinator_, *task_manager_, model_,
    task_mutex_, state_, backend_, [this]() {
      const int64_t now_ns = now().nanoseconds();
      const int64_t navigation_age = now_ns - navigator_received_ns_;
      const int64_t joints_age = now_ns - joints_received_ns_;
      return trajectory_action_slots_.empty() && state_.navigation_valid && state_.joints_valid &&
        (navigator_timeout_ <= 0 || (navigation_age >= 0 && navigation_age <= navigator_timeout_*1e9)) &&
        (joint_state_timeout_ <= 0 || (joints_age >= 0 && joints_age <= joint_state_timeout_*1e9));
    });
  manual_task_io_.configure(this, task_manager_->tasks(), task_mutex_,
    [this](const std::string & task, std::string & message) {return task_manager_->allow_mutation(task, message);});
  manual_task_io_.set_active(true);
  task_joint_target_subs_.clear();
  task_joint_trajectory_subs_.clear();
  task_joint_trajectory_action_servers_.clear();
  for (const auto & task : task_manager_->tasks()) {
    // Trajectory tasks expose their own typed interface; do not classify them
    // by a substring that can mistake base trajectory tasks for pose targets.
    if (std::dynamic_pointer_cast<BasePoseTrajectoryTask>(task)) {continue;}
    if (!std::dynamic_pointer_cast<PoseTaskCommon>(task) && task->build_status().target_type != "pose") {
      if (std::dynamic_pointer_cast<JointNominalTask>(task)) {
        const auto task_id = task->id();
        task_joint_target_subs_[task_id] =
          this->create_subscription<std_msgs::msg::Float64MultiArray>(
          "task_priority/tasks/" + task_id + "/joint_target",
          10,
          [this, task_id](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
            if (!msg) {
              return;
            }

            std::string message;
            std::scoped_lock lock(task_mutex_);
            if (!task_manager_->set_task_joint_target(task_id, msg->data, message)) {
              RCLCPP_WARN(
                this->get_logger(),
                "Rejected joint target for task '%s': %s",
                task_id.c_str(),
                message.c_str());
            }
          });
      } else if (std::dynamic_pointer_cast<JointTrajectoryTask>(task)) {
        const auto task_id = task->id();
        task_joint_trajectory_subs_[task_id] =
          this->create_subscription<trajectory_msgs::msg::JointTrajectory>(
          "task_priority/tasks/" + task_id + "/joint_trajectory",
          10,
          [this, task_id](const trajectory_msgs::msg::JointTrajectory::SharedPtr msg) {
            if (!msg) {
              return;
            }

            std::string message;
            std::scoped_lock lock(task_mutex_);
            if (cartesian_coordinator_.reserved(task_id) || !trajectory_task_enabled(task_id) || trajectory_action_slots_.count(task_id)) {
              RCLCPP_WARN(this->get_logger(),
                "Rejected trajectory topic for task '%s': disabled or owned by an action",
                task_id.c_str());
              return;
            }
            if (!task_manager_->set_task_joint_trajectory(task_id, *msg, message)) {
              RCLCPP_WARN(
                this->get_logger(),
                "Rejected joint trajectory for task '%s': %s",
                task_id.c_str(),
                message.c_str());
            }
          });

        task_joint_trajectory_action_servers_[task_id] =
          rclcpp_action::create_server<control_msgs::action::FollowJointTrajectory>(
          this,
          "task_priority/tasks/" + task_id + "/follow_joint_trajectory",
          [this, task_id](const rclcpp_action::GoalUUID & uuid,
          std::shared_ptr<const control_msgs::action::FollowJointTrajectory::Goal>)
          {
            std::scoped_lock lock(task_mutex_);
            if (cartesian_coordinator_.reserved(task_id) || !trajectory_task_enabled(task_id) || trajectory_action_slots_.count(task_id)) {
              return rclcpp_action::GoalResponse::REJECT;
            }
            trajectory_action_slots_.emplace(task_id, TrajectoryActionSlot{uuid, nullptr});
            return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
          },
          [this, task_id](
            const std::shared_ptr<rclcpp_action::ServerGoalHandle<
              control_msgs::action::FollowJointTrajectory>> handle)
          {
            std::string message;
            std::scoped_lock lock(task_mutex_);
            const auto slot = trajectory_action_slots_.find(task_id);
            if (slot == trajectory_action_slots_.end() || slot->second.uuid != handle->get_goal_id()) {
              return rclcpp_action::CancelResponse::REJECT;
            }
            task_manager_->cancel_task_joint_trajectory(task_id, message);
            return rclcpp_action::CancelResponse::ACCEPT;
          },
          [this, task_id](
            const std::shared_ptr<rclcpp_action::ServerGoalHandle<
              control_msgs::action::FollowJointTrajectory>> goal_handle)
          {
            execute_task_trajectory_goal(task_id, goal_handle);
          });
      }
      continue;
    }

    const auto task_id = task->id();
    task_target_subs_[task_id] = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "task_priority/tasks/" + task_id + "/target",
      10,
      [this, task_id](const geometry_msgs::msg::PoseStamped::SharedPtr msg,
        const rclcpp::MessageInfo & info) {
        if (!msg || pose_task_io_.is_own_target(task_id, info)) {
          return;
        }

        std::string message;
        std::scoped_lock lock(task_mutex_);
        if (!task_manager_->set_task_pose_goal(task_id, *msg, message)) {
          RCLCPP_WARN(
            this->get_logger(),
            "Rejected pose target for task '%s': %s",
            task_id.c_str(),
            message.c_str());
        }
      });
  }
  trajectory_result_timer_ = create_wall_timer(
    std::chrono::milliseconds(20), [this]() {poll_task_trajectory_goals();});
}

void RuntimeNode::configure_services()
{
  set_task_enabled_srv_ = this->create_service<srv::SetTaskEnabled>(
    "set_task_enabled",
    [this](
      const std::shared_ptr<srv::SetTaskEnabled::Request> request,
      std::shared_ptr<srv::SetTaskEnabled::Response> response) {
      std::scoped_lock lock(task_mutex_);
      response->success = task_manager_->set_task_enabled(request->task_id, request->enabled, response->message);
    });
  set_task_joint_activation_srv_ = this->create_service<srv::SetTaskJointActivation>(
    "set_task_joint_activation",
    [this](
      const std::shared_ptr<srv::SetTaskJointActivation::Request> request,
      std::shared_ptr<srv::SetTaskJointActivation::Response> response) {
      std::scoped_lock lock(task_mutex_);
      response->success = task_manager_->set_task_joint_activation(
        request->task_id,
        request->joint_activation,
        response->message);
    });
  reorder_tasks_srv_ = this->create_service<srv::ReorderTasks>(
    "reorder_tasks",
    [this](
      const std::shared_ptr<srv::ReorderTasks::Request> request,
      std::shared_ptr<srv::ReorderTasks::Response> response) {
      std::scoped_lock lock(task_mutex_);
      response->success = task_manager_->reorder_tasks(request->ordered_task_ids, response->message);
    });
  list_tasks_srv_ = this->create_service<srv::ListTasks>(
    "list_tasks",
    [this](
      const std::shared_ptr<srv::ListTasks::Request>,
      std::shared_ptr<srv::ListTasks::Response> response) {
      std::scoped_lock lock(task_mutex_);
      response->tasks = task_manager_->get_task_statuses();
      response->backend_name = backend_->name();
      response->solver_method = solver_method_name(parse_solver_method(this->get_parameter("solver_method").as_string()));
      response->success = true;
      response->message = "ok";
    });
  switch_backend_srv_ = this->create_service<srv::SwitchBackend>(
    "switch_backend",
    [this](
      const std::shared_ptr<srv::SwitchBackend::Request> request,
      std::shared_ptr<srv::SwitchBackend::Response> response) {
      std::scoped_lock lock(task_mutex_);
      if (!task_manager_->allow_mutation("*", response->message)) {response->success = false; return;}
      try {
        auto candidate = backend_loader_.createSharedInstance(request->backend_name);
        candidate->configure(*model_, resolve_robot_description(), get_logger());
        candidate->update(state_);
        for (const auto & task : task_manager_->tasks()) {
          auto collision = std::dynamic_pointer_cast<SelfCollisionAvoidanceTask>(task);
          if (!collision) {continue;}
          FrameState frame;
          for (const auto & capsule : collision->capsules()) {
            for (const auto & name : {collision->base_frame(), capsule.start_frame, capsule.end_frame}) {
              if (!candidate->read_frame_state(name, frame) || frame.jacobian.rows() != 6 ||
                frame.jacobian.cols() != static_cast<Eigen::Index>(model_->total_dofs()) ||
                !frame.jacobian.allFinite() || !frame.pose.matrix().allFinite()) {
                throw std::runtime_error("Backend cannot evaluate configured collision frame: " + name);
              }
            }
          }
        }
        backend_ = candidate;
        backend_plugin_name_ = request->backend_name;
        response->success = true;
        response->message = "Backend switched";
      } catch (const std::exception & ex) {
        response->success = false;
        response->message = ex.what();
      }
    });
  reset_srv_ = this->create_service<std_srvs::srv::Trigger>(
    "reset_solver",
    [this](
      const std::shared_ptr<std_srvs::srv::Trigger::Request>,
      std::shared_ptr<std_srvs::srv::Trigger::Response> response) {
      std::scoped_lock lock(task_mutex_);
      cartesian_execution_io_.stop_locked("Task priority runtime reset");
      task_manager_->reset();
      response->success = true;
      response->message = "Solver state reset";
  });
}

bool RuntimeNode::trajectory_task_enabled(const std::string & task_id) const
{
  for (const auto & task : task_manager_->tasks()) {
    if (task->id() == task_id) {return task->enabled();}
  }
  return false;
}

void RuntimeNode::execute_task_trajectory_goal(
  const std::string & task_id,
  const std::shared_ptr<TrajectoryGoalHandle> goal_handle)
{
  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
  std::string error;
  {
    std::scoped_lock lock(task_mutex_);
    const auto slot = trajectory_action_slots_.find(task_id);
    if (slot == trajectory_action_slots_.end() || slot->second.uuid != goal_handle->get_goal_id()) {
      error = "Trajectory admission expired";
    } else if (goal_handle->is_canceling()) {
      error = "Joint trajectory canceled before installation";
      trajectory_action_slots_.erase(slot);
    } else if (!trajectory_task_enabled(task_id)) {
      error = "Joint trajectory task is disabled";
      trajectory_action_slots_.erase(slot);
    } else if (!task_manager_->set_task_joint_trajectory(
        task_id, goal_handle->get_goal()->trajectory, error))
    {
      trajectory_action_slots_.erase(slot);
    } else {
      slot->second.handle = goal_handle;
      return;
    }
  }
  auto result = std::make_shared<FollowJointTrajectory::Result>();
  result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
  result->error_string = error;
  if (goal_handle->is_canceling()) {goal_handle->canceled(result);}
  else {goal_handle->abort(result);}
}

void RuntimeNode::poll_task_trajectory_goals()
{
  using FollowJointTrajectory = control_msgs::action::FollowJointTrajectory;
  std::vector<std::function<void()>> completions;
  {
    std::scoped_lock lock(task_mutex_);
    for (auto slot = trajectory_action_slots_.begin(); slot != trajectory_action_slots_.end();) {
      const auto handle = slot->second.handle;
      if (!handle) {++slot; continue;}
      const auto & task_id = slot->first;
      auto result = std::make_shared<FollowJointTrajectory::Result>();
      const bool canceling = handle->is_canceling();
      const bool ready = trajectory_task_enabled(task_id);
      const auto status = task_manager_->get_task_joint_trajectory_status(task_id);
      if (canceling || !ready) {
        std::string message;
        task_manager_->cancel_task_joint_trajectory(task_id, message);
        result->error_code = canceling ? FollowJointTrajectory::Result::SUCCESSFUL :
          FollowJointTrajectory::Result::INVALID_GOAL;
        result->error_string = canceling ? "Joint trajectory task canceled" :
          "Joint trajectory task is disabled";
      } else if (status.succeeded) {
        result->error_code = FollowJointTrajectory::Result::SUCCESSFUL;
        result->error_string = status.message;
      } else if (status.canceled || !status.accepted) {
        result->error_code = FollowJointTrajectory::Result::INVALID_GOAL;
        result->error_string = "Joint trajectory was cleared or disabled";
      } else if (status.timed_out) {
        result->error_code = FollowJointTrajectory::Result::GOAL_TOLERANCE_VIOLATED;
        result->error_string = status.message;
      } else {++slot; continue;}
      const bool succeeded = ready && !canceling && status.succeeded;
      completions.emplace_back([handle, result, canceling, succeeded]() {
        if (canceling) {handle->canceled(result);}
        else if (succeeded) {handle->succeed(result);}
        else {handle->abort(result);}
      });
      slot = trajectory_action_slots_.erase(slot);
    }
  }
  for (const auto & complete : completions) {complete();}
}

void RuntimeNode::configure_timer()
{
  const double rate_hz = std::max(1.0, this->get_parameter("rate_hz").as_double());
  timer_ = this->create_wall_timer(
    std::chrono::duration<double>(1.0 / rate_hz),
    std::bind(&RuntimeNode::timer_callback, this));
}

void RuntimeNode::navigator_callback(const sura_msgs::msg::Navigator::SharedPtr msg)
{
  std::scoped_lock lock(task_mutex_);
  if (!msg) {return;}
  const Eigen::Vector3d position(msg->position.position.x, msg->position.position.y, msg->position.position.z);
  const Eigen::Vector3d rpy(msg->rpy.x, msg->rpy.y, msg->rpy.z);
  state_.navigation_valid = position.allFinite() && rpy.allFinite();
  if (!state_.navigation_valid) {return;}
  state_.base_position = position; state_.base_rpy = rpy;
  navigator_received_ns_ = now().nanoseconds();
}

void RuntimeNode::joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  std::scoped_lock lock(task_mutex_);
  if (!msg) {return;}
  const auto all_joints = model_->all_joint_names();
  Eigen::VectorXd position = Eigen::VectorXd::Zero(all_joints.size());
  Eigen::VectorXd velocity = position;
  for (size_t i = 0; i < all_joints.size(); ++i) {
    const auto it = std::find(msg->name.begin(), msg->name.end(), all_joints[i]);
    if (it == msg->name.end()) {state_.joints_valid = false; return;}
    const size_t index = static_cast<size_t>(std::distance(msg->name.begin(), it));
    if (index >= msg->position.size() || !std::isfinite(msg->position[index])) {state_.joints_valid = false; return;}
    position(i) = msg->position[index];
    if (index < msg->velocity.size()) {velocity(i) = msg->velocity[index];}
  }
  if (!velocity.allFinite()) {state_.joints_valid = false; return;}
  state_.joint_positions = std::move(position); state_.joint_velocities = std::move(velocity);
  state_.joints_valid = true; joints_received_ns_ = now().nanoseconds();
}

void RuntimeNode::timer_callback()
{
  std::scoped_lock lock(task_mutex_);
  const int64_t now_ns = now().nanoseconds();
  const int64_t navigation_age = now_ns - navigator_received_ns_;
  const int64_t joints_age = now_ns - joints_received_ns_;
  const bool navigation_fresh = navigator_timeout_ <= 0 || (navigation_age >= 0 && navigation_age <= navigator_timeout_*1e9);
  const bool joints_fresh = joint_state_timeout_ <= 0 || (joints_age >= 0 && joints_age <= joint_state_timeout_*1e9);
  if (!state_.navigation_valid || !state_.joints_valid || !navigation_fresh || !joints_fresh) {
    if (cartesian_coordinator_.status().running) {
      cartesian_coordinator_.cancel_to_measured(state_, "Whole-body state is stale or unavailable", false);
    }
    geometry_msgs::msg::TwistStamped base_zero;
    base_zero.header.stamp = now(); base_zero.header.frame_id = model_->world_frame();
    base_command_pub_->publish(base_zero);
    std_msgs::msg::Float64MultiArray left_zero, right_zero;
    left_zero.data.assign(model_->left_arm_dofs(), 0.0);
    right_zero.data.assign(model_->right_arm_dofs(), 0.0);
    left_arm_command_pub_->publish(left_zero); right_arm_command_pub_->publish(right_zero);
    publish_status();
    return;
  }

  state_.ros_time_ns = now().nanoseconds();
  state_.ros_time_valid = true;
  backend_->update(state_);
  const auto & task_outputs = task_manager_->update_all_into(state_, *backend_);
  if (cartesian_coordinator_.status().running && task_manager_->arm_motion_stopped()) {
    cartesian_coordinator_.cancel_to_measured(state_, "A safety task stopped arm motion", false);
    task_manager_->update_all_into(state_, *backend_);
  }
  if (cartesian_coordinator_.update(state_, *backend_)) {
    task_manager_->update_all_into(state_, *backend_);
  }
  pose_task_io_.publish_targets(task_manager_->tasks(), task_outputs, now(), model_->world_frame());
  const auto tasks = task_manager_->tasks();
  for (size_t i = 0; i < tasks.size() && i < task_outputs.size(); ++i) {
    const auto & output = task_outputs[i];
    msg::TaskState message;
    message.id = tasks[i]->id();
    message.type = tasks[i]->plugin_name();
    message.enabled = tasks[i]->enabled();
    message.active = output.active;
    message.has_frame_pose = output.has_frame_pose;
    message.frame_id = output.frame_id;
    if (output.has_frame_pose) {
      const auto & p = output.frame_pose.translation();
      const Eigen::Quaterniond q(output.frame_pose.rotation());
      message.frame_pose.position.x = p.x();
      message.frame_pose.position.y = p.y();
      message.frame_pose.position.z = p.z();
      message.frame_pose.orientation.x = q.x();
      message.frame_pose.orientation.y = q.y();
      message.frame_pose.orientation.z = q.z();
      message.frame_pose.orientation.w = q.w();
    }
    const auto copy = [](const Eigen::VectorXd & values, std::vector<double> & target) {
      target.resize(static_cast<size_t>(values.size()));
      for (Eigen::Index j = 0; j < values.size(); ++j) {target[static_cast<size_t>(j)] = values(j);}
    };
    copy(output.feedforward, message.feedforward);
    copy(output.error, message.error);
    copy(output.desired_velocity, message.velocity);
    message.target = tasks[i]->current_target();
    task_state_pubs_.at(tasks[i]->id())->publish(message);
  }
  WholeBodyCommand command = solver_.solve(task_outputs);
  if (cartesian_coordinator_.observe_command(command, state_, *backend_)) {
    task_manager_->update_all_into(state_, *backend_);
    command = solver_.solve(task_outputs);
  }
  task_manager_->finalize_command(command, this->now().nanoseconds());
  command.base_velocity = command.generalized_velocity.head(model_->base_dofs());
  command.left_arm_velocity = command.generalized_velocity.segment(model_->left_offset(), model_->left_arm_dofs());
  command.right_arm_velocity = command.generalized_velocity.segment(model_->right_offset(), model_->right_arm_dofs());

  geometry_msgs::msg::TwistStamped base_msg;
  base_msg.header.stamp = this->now();
  base_msg.header.frame_id = model_->world_frame();
  if (command.base_velocity.size() >= 1) { base_msg.twist.linear.x = command.base_velocity(0); }
  if (command.base_velocity.size() >= 2) { base_msg.twist.linear.y = command.base_velocity(1); }
  if (command.base_velocity.size() >= 3) { base_msg.twist.linear.z = command.base_velocity(2); }
  if (command.base_velocity.size() >= 4) { base_msg.twist.angular.x = command.base_velocity(3); }
  if (command.base_velocity.size() >= 5) { base_msg.twist.angular.y = command.base_velocity(4); }
  if (command.base_velocity.size() >= 6) { base_msg.twist.angular.z = command.base_velocity(5); }
  base_command_pub_->publish(base_msg);

  std_msgs::msg::Float64MultiArray left_msg;
  left_msg.data.resize(command.left_arm_velocity.size());
  for (Eigen::Index i = 0; i < command.left_arm_velocity.size(); ++i) {
    left_msg.data[static_cast<size_t>(i)] = command.left_arm_velocity(i);
  }
  left_arm_command_pub_->publish(left_msg);

  std_msgs::msg::Float64MultiArray right_msg;
  right_msg.data.resize(command.right_arm_velocity.size());
  for (Eigen::Index i = 0; i < command.right_arm_velocity.size(); ++i) {
    right_msg.data[static_cast<size_t>(i)] = command.right_arm_velocity(i);
  }
  right_arm_command_pub_->publish(right_msg);

  publish_status();
}

void RuntimeNode::publish_status()
{
  msg::HierarchyState hierarchy;
  hierarchy.header.stamp = this->now();
  hierarchy.backend_name = backend_ ? backend_->name() : "none";
  hierarchy.solver_method = this->get_parameter("solver_method").as_string();
  hierarchy.ready = state_.navigation_valid && state_.joints_valid;
  hierarchy.tasks = task_manager_->get_task_statuses();
  hierarchy_pub_->publish(hierarchy);

  msg::SolverDiagnostics diagnostics;
  diagnostics.header = hierarchy.header;
  diagnostics.backend_name = hierarchy.backend_name;
  diagnostics.solver_method = hierarchy.solver_method;
  diagnostics.damped_least_squares_lambda = this->get_parameter("dls_lambda").as_double();
  diagnostics.condition_estimate = solver_.last_condition_estimate();
  diagnostics.state_valid = hierarchy.ready;
  diagnostics.message = hierarchy.ready ? "running" : "waiting_for_state";
  diagnostics_pub_->publish(diagnostics);
}

rcl_interfaces::msg::SetParametersResult RuntimeNode::on_parameters_set(
  const std::vector<rclcpp::Parameter> & parameters)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  result.reason = "accepted";
  std::unique_lock<std::mutex> task_lock(task_mutex_, std::try_to_lock);
  if (!task_lock.owns_lock()) {
    result.successful = false;
    result.reason = "A controller update is in progress; retry the parameter change";
    return result;
  }
  const auto collision_validation = validate_collision_parameter_updates(parameters, task_manager_->tasks(), false);
  if (!collision_validation.successful) {return collision_validation;}
  const auto limits_validation = validate_joint_limits_marker_parameter_updates(parameters, task_manager_->tasks(), false);
  if (!limits_validation.successful) {return limits_validation;}

  for (const auto & parameter : parameters) {
    if (parameter.get_name() == "dls_lambda") {
      if (parameter.as_double() <= 0.0) {
        result.successful = false;
        result.reason = "dls_lambda must be positive";
        return result;
      }
      solver_.set_damping(parameter.as_double());
    } else if (parameter.get_name() == "solver_method") {
      solver_.set_method(parse_solver_method(parameter.as_string()));
    } else if (parameter.get_name() == "dof_weights") {
      const auto values = parameter.as_double_array();
      Eigen::VectorXd w(values.size());
      for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] <= 0.0) {
          result.successful = false;
          result.reason = "dof_weights must be positive";
          return result;
        }
        w(static_cast<Eigen::Index>(i)) = values[i];
      }
      solver_.set_weights(w);
    } else if (parameter.get_name() == "velocity_limits") {
      const auto values = parameter.as_double_array();
      Eigen::VectorXd limits(values.size());
      for (size_t i = 0; i < values.size(); ++i) {
        if (values[i] <= 0.0) {
          result.successful = false;
          result.reason = "velocity_limits must be positive";
          return result;
        }
        limits(static_cast<Eigen::Index>(i)) = values[i];
      }
      solver_.set_velocity_limits(limits);
    } else {
      std::string task_id;
      std::string field;
      if (!split_task_parameter_name(parameter.get_name(), task_id, field)) {
        continue;
      }

      std::string message;
      if (field == "auto_update_target_pose" || field == "feedforward_velocity_threshold") {
        if (!task_manager_->set_task_pose_feedforward_option(task_id, field, parameter, message)) {
          result.successful = false;
          result.reason = message;
          return result;
        }
      } else if (field == "enabled") {
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL ||
          !task_manager_->set_task_enabled(task_id, parameter.as_bool(), message)) {
          result.successful = false; result.reason = "Invalid task enabled parameter"; return result;
        }
      } else if (field == "gain") {
        const auto values = parameter.as_double_array();
        if (!values_are_non_negative(values)) {
          result.successful = false;
          result.reason = parameter.get_name() + " values must be non-negative";
          return result;
        }
        if (!task_manager_->set_task_gain(task_id, values, message)) {
          result.successful = false;
          result.reason = message;
          return result;
        }
      } else if (field == "gain_scalar") {
        const double value = parameter.as_double();
        if (value < 0.0) {
          result.successful = false;
          result.reason = parameter.get_name() + " must be non-negative";
          return result;
        }
        if (!task_manager_->set_task_gain_scalar(task_id, value, message)) {
          result.successful = false;
          result.reason = message;
          return result;
        }
      }
    }
  }

  validate_collision_parameter_updates(parameters, task_manager_->tasks(), true);
  validate_joint_limits_marker_parameter_updates(parameters, task_manager_->tasks(), true);
  return result;
}

}  // namespace task_priority_kinematic_control
