#include "task_priority_kinematic_control/core/runtime_node.hpp"

#include <rcl_interfaces/msg/set_parameters_result.hpp>
#include <algorithm>
#include <chrono>
#include <stdexcept>

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
  task_manager_->configure(TaskContext{model_});
  configure_publishers();
  configure_subscribers();
  configure_services();
  configure_timer();
  param_callback_handle_ = this->add_on_set_parameters_callback(
    std::bind(&RuntimeNode::on_parameters_set, this, std::placeholders::_1));
}

void RuntimeNode::declare_parameters()
{
  declare_if_missing<std::string>(*this, "backend_plugin", "task_priority_kinematic_control/KDLKinematicsBackend");
  declare_if_missing<std::string>(*this, "robot_description", "");
  declare_if_missing<std::string>(*this, "robot_description_source_node", "/robot_state_publisher");
  declare_if_missing<double>(*this, "robot_description_wait_timeout_sec", 3.0);
  declare_if_missing<std::string>(*this, "world_frame", "world_ned");
  declare_if_missing<std::string>(*this, "base_frame", "cirtesub/base_link");
  declare_if_missing<std::string>(*this, "left_tip_frame", "cirtesub/alpha_left/link6");
  declare_if_missing<std::string>(*this, "right_tip_frame", "cirtesub/alpha_right/link6");
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
  declare_if_missing<std::string>(*this, "left_arm_command_topic", "/cirtesub/controller/alpha_left_forward_velocity_controller/commands");
  declare_if_missing<std::string>(*this, "right_arm_command_topic", "/cirtesub/controller/alpha_right_forward_velocity_controller/commands");
  declare_if_missing<std::vector<std::string>>(*this, "task_ids", {});
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
  navigator_sub_ = this->create_subscription<sura_msgs::msg::Navigator>(
    this->get_parameter("navigator_topic").as_string(),
    10,
    std::bind(&RuntimeNode::navigator_callback, this, std::placeholders::_1));
  joint_state_sub_ = this->create_subscription<sensor_msgs::msg::JointState>(
    this->get_parameter("joint_states_topic").as_string(),
    10,
    std::bind(&RuntimeNode::joint_state_callback, this, std::placeholders::_1));

  task_target_subs_.clear();
  task_joint_target_subs_.clear();
  for (const auto & task : task_manager_->tasks()) {
    if (task->plugin_name().find("PoseTask") == std::string::npos) {
      if (task->plugin_name().find("JointNominalTask") != std::string::npos) {
        const auto task_id = task->id();
        task_joint_target_subs_[task_id] =
          this->create_subscription<std_msgs::msg::Float64MultiArray>(
          "/cirtesub/controller/task_priority/tasks/" + task_id + "/joint_target",
          10,
          [this, task_id](const std_msgs::msg::Float64MultiArray::SharedPtr msg) {
            if (!msg) {
              return;
            }

            std::string message;
            if (!task_manager_->set_task_joint_target(task_id, msg->data, message)) {
              RCLCPP_WARN(
                this->get_logger(),
                "Rejected joint target for task '%s': %s",
                task_id.c_str(),
                message.c_str());
            }
          });
      }
      continue;
    }

    const auto task_id = task->id();
    task_target_subs_[task_id] = this->create_subscription<geometry_msgs::msg::PoseStamped>(
      "/cirtesub/controller/task_priority/tasks/" + task_id + "/target",
      10,
      [this, task_id](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
        if (!msg) {
          return;
        }

        std::string message;
        if (!task_manager_->set_task_pose_goal(task_id, *msg, message)) {
          RCLCPP_WARN(
            this->get_logger(),
            "Rejected pose target for task '%s': %s",
            task_id.c_str(),
            message.c_str());
        }
      });
  }
}

void RuntimeNode::configure_services()
{
  set_task_enabled_srv_ = this->create_service<srv::SetTaskEnabled>(
    "set_task_enabled",
    [this](
      const std::shared_ptr<srv::SetTaskEnabled::Request> request,
      std::shared_ptr<srv::SetTaskEnabled::Response> response) {
      response->success = task_manager_->set_task_enabled(request->task_id, request->enabled, response->message);
    });
  reorder_tasks_srv_ = this->create_service<srv::ReorderTasks>(
    "reorder_tasks",
    [this](
      const std::shared_ptr<srv::ReorderTasks::Request> request,
      std::shared_ptr<srv::ReorderTasks::Response> response) {
      response->success = task_manager_->reorder_tasks(request->ordered_task_ids, response->message);
    });
  list_tasks_srv_ = this->create_service<srv::ListTasks>(
    "list_tasks",
    [this](
      const std::shared_ptr<srv::ListTasks::Request>,
      std::shared_ptr<srv::ListTasks::Response> response) {
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
      try {
        backend_plugin_name_ = request->backend_name;
        rebuild_backend();
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
      task_manager_->reset();
      response->success = true;
      response->message = "Solver state reset";
    });
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
  state_.base_position = Eigen::Vector3d(
    msg->position.position.x,
    msg->position.position.y,
    msg->position.position.z);
  state_.base_rpy = Eigen::Vector3d(msg->rpy.x, msg->rpy.y, msg->rpy.z);
  state_.navigation_valid = true;
}

void RuntimeNode::joint_state_callback(const sensor_msgs::msg::JointState::SharedPtr msg)
{
  const auto all_joints = model_->all_joint_names();
  state_.joint_positions = Eigen::VectorXd::Zero(all_joints.size());
  state_.joint_velocities = Eigen::VectorXd::Zero(all_joints.size());
  for (size_t i = 0; i < all_joints.size(); ++i) {
    const auto it = std::find(msg->name.begin(), msg->name.end(), all_joints[i]);
    if (it == msg->name.end()) {
      continue;
    }
    const size_t idx = static_cast<size_t>(std::distance(msg->name.begin(), it));
    if (idx < msg->position.size()) {
      state_.joint_positions(static_cast<Eigen::Index>(i)) = msg->position[idx];
    }
    if (idx < msg->velocity.size()) {
      state_.joint_velocities(static_cast<Eigen::Index>(i)) = msg->velocity[idx];
    }
  }
  state_.joints_valid = true;
}

void RuntimeNode::timer_callback()
{
  if (!state_.navigation_valid || !state_.joints_valid) {
    publish_status();
    return;
  }

  backend_->update(state_);
  const auto task_outputs = task_manager_->update_all(state_, *backend_);
  WholeBodyCommand command = solver_.solve(task_outputs);
  command.base_velocity = command.generalized_velocity.head(model_->base_dofs());
  command.left_arm_velocity = command.generalized_velocity.segment(model_->left_offset(), model_->left_arm_dofs());
  command.right_arm_velocity = command.generalized_velocity.segment(model_->right_offset(), model_->right_arm_dofs());

  geometry_msgs::msg::TwistStamped base_msg;
  base_msg.header.stamp = this->now();
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
    }
  }

  return result;
}

}  // namespace task_priority_kinematic_control
