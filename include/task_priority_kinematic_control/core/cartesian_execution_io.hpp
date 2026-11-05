#pragma once

#include "task_priority_kinematic_control/core/cartesian_trajectory_coordinator.hpp"
#include <rclcpp/logging.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <optional>

namespace task_priority_kinematic_control
{

// Shared admission/lifecycle for the standalone runtime and ros2_control host.
class CartesianExecutionIO
{
public:
  using Action = CartesianTrajectoryCoordinator::Action;
  using Handle = rclcpp_action::ServerGoalHandle<Action>;
  template<class NodeT>
  static void declare_parameters(NodeT * node)
  {
    const auto declare_string = [&](const std::string & name, const std::string & value) {
      if (!node->has_parameter(name)) {node->template declare_parameter<std::string>(name, value);}
    };
    declare_string("cartesian_execution.base_task", "base_pose_trajectory");
    declare_string("cartesian_execution.left_joint_task", "left_joint_trajectory");
    declare_string("cartesian_execution.right_joint_task", "right_joint_trajectory");
    declare_string("cartesian_execution.left_hold_task", "left_arm_nominal");
    declare_string("cartesian_execution.right_hold_task", "right_arm_nominal");
    if (!node->has_parameter("cartesian_execution.safety_tasks")) {
      node->template declare_parameter<std::vector<std::string>>("cartesian_execution.safety_tasks",
        {"joint_limits", "self_collision_avoidance", "stabilize_limits"});
    }
    if (!node->has_parameter("cartesian_execution.other_arm_tolerance")) {
      node->template declare_parameter<double>("cartesian_execution.other_arm_tolerance", 0.05);
    }
  }
  template<class NodeT>
  void configure(NodeT * node, CartesianTrajectoryCoordinator & coordinator,
    TaskManager & manager, const std::shared_ptr<WholeBodyModel> & model,
    std::mutex & mutex, WholeBodyState & state, KinematicsBackendPtr & backend,
    std::function<bool()> ready)
  {
    declare_parameters(node);
    CartesianTrajectoryCoordinator::Settings settings;
    settings.base_task = node->get_parameter("cartesian_execution.base_task").as_string();
    settings.left_joint_task = node->get_parameter("cartesian_execution.left_joint_task").as_string();
    settings.right_joint_task = node->get_parameter("cartesian_execution.right_joint_task").as_string();
    settings.left_hold_task = node->get_parameter("cartesian_execution.left_hold_task").as_string();
    settings.right_hold_task = node->get_parameter("cartesian_execution.right_hold_task").as_string();
    settings.safety_tasks = node->get_parameter("cartesian_execution.safety_tasks").as_string_array();
    settings.other_arm_tolerance = node->get_parameter("cartesian_execution.other_arm_tolerance").as_double();
    coordinator.configure(manager, model, node->get_clock(), settings);
    coordinator_ = &coordinator; mutex_ = &mutex; state_ = &state; backend_ = &backend;
    manager.set_mutation_guard([&coordinator, &state, &backend](
      const std::string & task, std::string & message) {
        return !backend || coordinator.allow_mutation(task, state, *backend, message);
      });
    server_ = rclcpp_action::create_server<Action>(node,
      "task_priority/follow_cartesian_trajectory",
      [this, ready, logger = node->get_logger()](
        const rclcpp_action::GoalUUID & uuid, std::shared_ptr<const Action::Goal> goal) {
        std::scoped_lock lock(*mutex_);
        std::string message;
        if (admission_ || handle_) {
          message = "Another coordinated trajectory is already admitted or executing";
        } else if (!ready()) {
          message = "Controller is not ready: check active state, fresh navigation and competing trajectory actions";
        } else if (!*backend_) {
          message = "Kinematics backend is unavailable";
        } else if (coordinator_->validate_goal(*goal, *state_, **backend_, message)) {
          admission_ = uuid;
          return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
        }
        RCLCPP_WARN(logger, "Coordinated Cartesian execution rejected: %s", message.c_str());
        return rclcpp_action::GoalResponse::REJECT;
      },
      [this](const std::shared_ptr<Handle> handle) {
        std::scoped_lock lock(*mutex_);
        if (handle_ && handle_->get_goal_id() == handle->get_goal_id()) {
          coordinator_->cancel_to_measured(*state_, "Cartesian trajectory canceled");
          return rclcpp_action::CancelResponse::ACCEPT;
        }
        if (admission_ && *admission_ == handle->get_goal_id()) {
          admission_.reset(); return rclcpp_action::CancelResponse::ACCEPT;
        }
        return rclcpp_action::CancelResponse::REJECT;
      },
      [this, ready](const std::shared_ptr<Handle> handle) {
        std::scoped_lock lock(*mutex_);
        std::string message;
        if (!admission_ || *admission_ != handle->get_goal_id() || handle->is_canceling() ||
          !ready() || !*backend_ ||
          !coordinator_->install(*handle->get_goal(), *state_, **backend_, message))
        {
          admission_.reset();
          auto result = std::make_shared<Action::Result>();
          result->success = false; result->message = message.empty() ? "Cartesian trajectory admission expired" : message;
          if (handle->is_canceling()) {handle->canceled(result);} else {handle->abort(result);}
          return;
        }
        admission_.reset(); handle_ = handle;
      });
    release_ = node->template create_service<std_srvs::srv::Trigger>(
      "task_priority/release_cartesian_trajectory",
      [this](std_srvs::srv::Trigger::Request::SharedPtr,
        std_srvs::srv::Trigger::Response::SharedPtr response) {
        std::scoped_lock lock(*mutex_);
        admission_.reset();
        if (!*backend_) {response->success = false; response->message = "Kinematics backend unavailable"; return;}
        if (coordinator_->status().running) {
          coordinator_->cancel_to_measured(*state_, "Cartesian trajectory released by operator");
        }
        response->success = coordinator_->release(*state_, **backend_, response->message);
      });
    timer_ = node->create_wall_timer(std::chrono::milliseconds(20), [this]() {poll();});
  }

  void poll()
  {
    if (!mutex_ || !coordinator_) {return;}
    std::shared_ptr<Handle> complete;
    std::shared_ptr<Action::Result> result;
    bool canceled = false;
    {
      std::scoped_lock lock(*mutex_);
      if (!handle_) {return;}
      const auto & status = coordinator_->status();
      if (!status.complete) {
        auto feedback = std::make_shared<Action::Feedback>();
        feedback->progress = status.progress; feedback->current_stage = status.message;
        feedback->message = status.message;
        std::copy(status.base_error.data(), status.base_error.data()+6, feedback->base_error.begin());
        std::copy(status.tip_error.data(), status.tip_error.data()+6, feedback->tip_error.begin());
        feedback->joint_error.assign(status.joint_error.data(), status.joint_error.data()+status.joint_error.size());
        handle_->publish_feedback(feedback);
        return;
      }
      result = std::make_shared<Action::Result>();
      result->success = status.success; result->message = status.message;
      canceled = handle_->is_canceling();
      complete = std::move(handle_);
    }
    if (canceled) {complete->canceled(result);}
    else if (result->success) {complete->succeed(result);}
    else {complete->abort(result);}
  }

  // Caller already holds the host mutex; release before disabling/resetting all tasks.
  void stop_locked(const std::string & reason)
  {
    admission_.reset();
    if (coordinator_ && backend_ && *backend_) {coordinator_->shutdown(*state_, **backend_, reason);}
    if (handle_) {
      auto result = std::make_shared<Action::Result>(); result->success = false; result->message = reason;
      if (handle_->is_canceling()) {handle_->canceled(result);} else {handle_->abort(result);}
      handle_.reset();
    }
  }

private:
  CartesianTrajectoryCoordinator * coordinator_ = nullptr;
  std::mutex * mutex_ = nullptr;
  WholeBodyState * state_ = nullptr;
  KinematicsBackendPtr * backend_ = nullptr;
  std::optional<rclcpp_action::GoalUUID> admission_;
  std::shared_ptr<Handle> handle_;
  rclcpp_action::Server<Action>::SharedPtr server_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr release_;
  rclcpp::TimerBase::SharedPtr timer_;
};

}  // namespace task_priority_kinematic_control
