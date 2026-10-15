#pragma once

#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>
#include <mutex>
#include <algorithm>

namespace task_priority_kinematic_control
{
// All callbacks and the solver share the host's task mutex.
class ManualTaskIO
{
public:
  template<class NodeT>
  void configure(NodeT * node, const std::vector<std::shared_ptr<TaskBase>> & tasks,
    std::mutex & task_mutex)
  {
    services_.clear();
    subscriptions_.clear();
    nominals_.clear();
    active_ = false;
    for (const auto & task : tasks) {
      const std::string prefix = "task_priority/tasks/" + task->id();
      if (auto nominal = std::dynamic_pointer_cast<JointNominalTask>(task)) {
        nominals_.push_back(nominal);
        subscriptions_.push_back(node->template create_subscription<sensor_msgs::msg::JointState>(
          prefix + "/joint_command", rclcpp::QoS(1),
          [this, weak = std::weak_ptr<JointNominalTask>(nominal), &task_mutex,
          logger = node->get_logger()](const sensor_msgs::msg::JointState::SharedPtr msg) {
            if (!msg) {return;}
            std::scoped_lock lock(task_mutex);
            if (auto task = weak.lock()) {
              auto command = *msg;
              if (!active_) {std::fill(command.velocity.begin(), command.velocity.end(), 0.0);}
              std::string message;
              if (!task->set_joint_command(command, message)) {
                RCLCPP_WARN(logger, "Rejected joint command for '%s': %s", task->id().c_str(), message.c_str());
              }
            }
          }));
      }
      auto frame = std::dynamic_pointer_cast<FramePoseTask>(task);
      if (!frame || !frame->roll_pitch_control()) {continue;}
      for (const bool enable : {true, false}) {
        services_.push_back(node->template create_service<std_srvs::srv::Trigger>(
          prefix + (enable ? "/enable_roll_pitch" : "/disable_roll_pitch"),
          [weak = std::weak_ptr<FramePoseTask>(frame), &task_mutex, enable](
            std_srvs::srv::Trigger::Request::SharedPtr,
            std_srvs::srv::Trigger::Response::SharedPtr response) {
            std::scoped_lock lock(task_mutex);
            if (auto task = weak.lock()) {
              response->success = task->set_allow_roll_pitch(enable, response->message);
            } else {response->success = false; response->message = "Task unavailable";}
          }));
      }
    }
  }

  // Called with the host task mutex held, also on configure/deactivate/cleanup.
  void set_active(bool active)
  {
    active_ = active;
    for (const auto & weak : nominals_) {
      if (auto task = weak.lock()) {task->clear_velocity();}
    }
  }

private:
  bool active_ = false;
  std::vector<std::weak_ptr<JointNominalTask>> nominals_;
  std::vector<rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr> services_;
  std::vector<rclcpp::Subscription<sensor_msgs::msg::JointState>::SharedPtr> subscriptions_;
};
}  // namespace task_priority_kinematic_control
