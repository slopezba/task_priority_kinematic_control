#pragma once

#include "task_priority_kinematic_control/core/stpik_check.hpp"
#include <rclcpp_action/rclcpp_action.hpp>
#include <atomic>
#include <future>
#include <mutex>
#include <optional>
#include <algorithm>

namespace task_priority_kinematic_control
{
// ROS adapter only. The worker owns its snapshot and never accesses the real controller.
class StpikCheckIO
{
public:
  using Handle = rclcpp_action::ServerGoalHandle<StpikAction>;
  using Capture = std::function<StpikSnapshot()>;
  using Validate = std::function<bool(const StpikSnapshot &, const StpikOptions &, std::string &)>;
  ~StpikCheckIO()
  {
    try {stop();} catch (const std::exception &) {}
  }

  template<class NodeT>
  static void declare_parameters(NodeT * node)
  {
    const auto scalar = [node](const std::string & name, double value) {
      if (!node->has_parameter(name)) {node->template declare_parameter<double>(name, value);}
    };
    scalar("stpik.dt", 0.01); scalar("stpik.stall_window", 1.0);
    scalar("stpik.min_position_progress", 0.0001); scalar("stpik.min_orientation_progress", 0.001);
    scalar("stpik.numerical_tolerance", 1e-6); scalar("stpik.singular_value_tolerance", 1e-6);
    scalar("stpik.wall_timeout", 30.0); scalar("stpik.snapshot_joint_tolerance", 0.001);
    scalar("stpik.snapshot_position_tolerance", 0.005); scalar("stpik.snapshot_orientation_tolerance", 0.01);
    const auto array = [node](const std::string & name, const std::vector<double> & value) {
      if (!node->has_parameter(name)) {node->template declare_parameter<std::vector<double>>(name, value);}
    };
    array("stpik.velocity_limits", {});
  }

  template<class NodeT>
  static StpikOptions read_options(NodeT * node)
  {
    StpikOptions out;
    out.dt = node->get_parameter("stpik.dt").as_double();
    out.stall_window = node->get_parameter("stpik.stall_window").as_double();
    out.min_position_progress = node->get_parameter("stpik.min_position_progress").as_double();
    out.min_orientation_progress = node->get_parameter("stpik.min_orientation_progress").as_double();
    out.numerical_tolerance = node->get_parameter("stpik.numerical_tolerance").as_double();
    out.singular_value_tolerance = node->get_parameter("stpik.singular_value_tolerance").as_double();
    out.wall_timeout = node->get_parameter("stpik.wall_timeout").as_double();
    out.snapshot_joint_tolerance = node->get_parameter("stpik.snapshot_joint_tolerance").as_double();
    out.snapshot_position_tolerance = node->get_parameter("stpik.snapshot_position_tolerance").as_double();
    out.snapshot_orientation_tolerance = node->get_parameter("stpik.snapshot_orientation_tolerance").as_double();
    out.velocity_limits = node->get_parameter("stpik.velocity_limits").as_double_array();
    std::string error;
    if (!out.valid(error)) {throw std::invalid_argument(error);}
    return out;
  }

  template<class NodeT>
  void configure(NodeT * node, Capture capture, Validate validate)
  {
    stop();
    declare_parameters(node);
    capture_ = std::move(capture); validate_ = std::move(validate);
    options_ = [node]() {return read_options(node);};
    busy_.store(false);
    server_ = rclcpp_action::create_server<StpikAction>(node, "task_priority/check_cartesian_trajectory",
      [this](const rclcpp_action::GoalUUID &, std::shared_ptr<const StpikAction::Goal>) {
        bool expected = false;
        return busy_.compare_exchange_strong(expected, true) ?
          rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE : rclcpp_action::GoalResponse::REJECT;
      },
      [this](const std::shared_ptr<Handle>) {
        cancel_.store(true); return rclcpp_action::CancelResponse::ACCEPT;
      },
      [this](const std::shared_ptr<Handle> handle) {accept(handle);});
    timer_ = node->create_wall_timer(std::chrono::milliseconds(20), [this]() {poll();});
  }

  // May be called at lifecycle shutdown before the host destroys state/backend.
  void stop();
  void cancel_pending()
  {
    cancel_.store(true);
    std::scoped_lock lock(mutex_); validated_snapshot_.reset(); validated_id_ = 0;
  }

  template<class NodeT>
  void configure_host(NodeT * node, std::mutex & mutex, WholeBodyState & state,
    std::shared_ptr<WholeBodyModel> & model, KinematicsBackendPtr & backend,
    HierarchySolver & solver, std::unique_ptr<TaskManager> & manager,
    std::function<bool()> ready)
  {
    configure(node, [&mutex, &state, &model, &backend, &solver, &manager, ready]() {
        std::scoped_lock lock(mutex);
        if (!ready() || !manager || !backend || !model) {
          StpikSnapshot unavailable;
          unavailable.unavailable_reason = "Controller is inactive, busy, changing parameters or has stale state";
          return unavailable;
        }
        return StpikTaskAccess::capture(state, model, backend, solver, manager->tasks());
      }, [&mutex, &state, &model, &backend, &solver, &manager, ready](
        const StpikSnapshot & snapshot, const StpikOptions & settings, std::string & reason) {
        std::scoped_lock lock(mutex);
        if (!ready() || !manager || !model) {reason = "Controller is no longer ready for Pose execution"; return false;}
        return StpikTaskAccess::compatible(snapshot, state, *model, backend.get(), solver, manager->tasks(), settings, reason);
      });
  }

private:
  void accept(const std::shared_ptr<Handle> &);
  void poll();
  void complete(StpikAction::Result);
  static bool same_request(const StpikAction::Goal &, const StpikAction::Goal &);
  Capture capture_;
  Validate validate_;
  std::function<StpikOptions()> options_;
  std::atomic_bool busy_{false}, cancel_{false};
  std::atomic<double> progress_time_{0.0}, progress_total_{1.0};
  std::mutex mutex_;
  std::shared_ptr<Handle> handle_;
  std::future<StpikAction::Result> worker_;
  std::optional<StpikSnapshot> snapshot_, validated_snapshot_;
  StpikAction::Goal request_, validated_request_;
  StpikOptions settings_, validated_settings_;
  uint64_t next_id_ = 0, validated_id_ = 0;
  rclcpp_action::Server<StpikAction>::SharedPtr server_;
  rclcpp::TimerBase::SharedPtr timer_;
};
}  // namespace task_priority_kinematic_control
