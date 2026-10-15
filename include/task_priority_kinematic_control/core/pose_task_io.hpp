#pragma once

#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include <rclcpp/message_info.hpp>
#include <atomic>
#include <map>
#include <stdexcept>

namespace task_priority_kinematic_control
{
// A commanded velocity is evaluated at the controlled point. Only its axes
// change: moving the reference origin must not introduce an omega cross p term.
inline PoseVelocity rotate_pose_velocity(
  const geometry_msgs::msg::Twist & twist, const Eigen::Quaterniond & rotation)
{
  PoseVelocity result;
  result.head<3>() = rotation * Eigen::Vector3d(twist.linear.x, twist.linear.y, twist.linear.z);
  result.tail<3>() = rotation * Eigen::Vector3d(twist.angular.x, twist.angular.y, twist.angular.z);
  return result;
}

class PoseTaskIO
{
public:
  ~PoseTaskIO() {set_active(false);}

  template<class NodeT>
  void configure(NodeT * node, const std::vector<std::shared_ptr<TaskBase>> & tasks,
    const std::string & world_frame, const std::string & base_frame)
  {
    set_active(false);
    std::scoped_lock callback_lock(callback_mutex_);
    subscriptions_.clear();
    publishers_.clear();
    tasks_.clear();
    listener_.reset();
    buffer_ = std::make_unique<tf2_ros::Buffer>(node->get_clock());
    listener_ = std::make_unique<tf2_ros::TransformListener>(*buffer_);
    for (const auto & task : tasks) {
      auto pose_task = std::dynamic_pointer_cast<PoseTaskCommon>(task);
      if (!pose_task) {continue;}
      const std::string prefix = "task_priority/tasks/" + task->id();
      tasks_[task->id()] = pose_task;
      publishers_[task->id()] = node->template create_publisher<geometry_msgs::msg::PoseStamped>(
        prefix + "/target", rclcpp::SystemDefaultsQoS());
      subscriptions_[task->id()] =
        node->template create_subscription<geometry_msgs::msg::TwistStamped>(
        prefix + "/feedforward", rclcpp::QoS(1),
        [this, weak = std::weak_ptr<PoseTaskCommon>(pose_task),
        world_frame, base_frame, logger = node->get_logger()](
          const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
          std::scoped_lock callback_lock(callback_mutex_);
          auto task = weak.lock();
          if (!task || !msg || !active_.load()) {return;}
          const auto epoch = task->feedforward_epoch();
          PoseVelocity velocity = PoseVelocity::Zero();
          try {
            const auto & t = msg->twist;
            PoseVelocity input;
            input << t.linear.x, t.linear.y, t.linear.z, t.angular.x, t.angular.y, t.angular.z;
            if (!input.allFinite()) {throw std::invalid_argument("Non-finite feedforward");}
            const auto & frame = msg->header.frame_id.empty() ? base_frame : msg->header.frame_id;
            if (input.isZero(0.0) || frame == world_frame) {
              velocity = input;
            } else {
              const auto transform = buffer_->lookupTransform(
                world_frame, frame, rclcpp::Time(msg->header.stamp));
              const auto & q = transform.transform.rotation;
              Eigen::Quaterniond rotation(q.w, q.x, q.y, q.z);
              if (!rotation.coeffs().allFinite() || rotation.norm() < 1e-12) {
                throw std::invalid_argument("Invalid reference rotation");
              }
              velocity = rotate_pose_velocity(t, rotation.normalized());
              if (!velocity.allFinite()) {throw std::invalid_argument("Feedforward rotation overflow");}
            }
          } catch (const std::exception & error) {
            RCLCPP_WARN(logger, "Feedforward for '%s' cleared: %s", task->id().c_str(), error.what());
          }
          if (active_.load()) {task->set_feedforward(velocity, epoch);}
        });
    }
  }

  // The host holds its task lock when resetting or publishing computations.
  void set_active(bool active)
  {
    std::scoped_lock callback_lock(callback_mutex_);
    active_.store(false);
    for (const auto & entry : tasks_) {
      if (auto task = entry.second.lock()) {task->reset();}
    }
    active_.store(active);
  }

  bool is_own_target(const std::string & task_id, const rclcpp::MessageInfo & info) const
  {
    std::scoped_lock callback_lock(callback_mutex_);
    const auto publisher = publishers_.find(task_id);
    return publisher != publishers_.end() &&
      *publisher->second == info.get_rmw_message_info().publisher_gid;
  }

  void publish_targets(const std::vector<std::shared_ptr<TaskBase>> & tasks,
    const std::vector<TaskComputation> & outputs, const rclcpp::Time & time,
    const std::string & world_frame)
  {
    for (size_t i = 0; i < tasks.size() && i < outputs.size(); ++i) {
      const auto publisher = publishers_.find(tasks[i]->id());
      if (publisher == publishers_.end() || !outputs[i].has_updated_target) {continue;}
      geometry_msgs::msg::PoseStamped target;
      target.header.stamp = time;
      target.header.frame_id = world_frame;
      target.pose = outputs[i].updated_target;
      publisher->second->publish(target);
    }
  }

private:
  friend class PoseTaskIOTest;

  mutable std::mutex callback_mutex_;
  std::atomic_bool active_{false};
  std::unique_ptr<tf2_ros::Buffer> buffer_;
  std::unique_ptr<tf2_ros::TransformListener> listener_;
  std::map<std::string, std::weak_ptr<PoseTaskCommon>> tasks_;
  std::map<std::string, rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr> subscriptions_;
  std::map<std::string, rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr> publishers_;
};
}  // namespace task_priority_kinematic_control
