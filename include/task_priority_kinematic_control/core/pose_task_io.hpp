#pragma once

#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"
#include "geometry_msgs/msg/twist_stamped.hpp"
#include "std_msgs/msg/bool.hpp"
#include "tf2_ros/buffer.h"
#include "tf2_ros/transform_listener.h"
#include <rclcpp/message_info.hpp>
#include <atomic>
#include <cmath>
#include <map>
#include <stdexcept>
#include <functional>

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
    const std::string & world_frame, const std::string & base_frame,
    std::mutex * host_mutex = nullptr,
    std::function<bool(const std::string &, std::string &)> mutation_guard = {})
  {
    set_active(false);
    std::scoped_lock callback_lock(callback_mutex_);
    subscriptions_.clear();
    publishers_.clear();
    offset_subscriptions_.clear();
    use_offset_publishers_.clear();
    published_use_offset_.clear();
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
      // Offset of the controlled point (e.g. zip-tie tip) in the task frame. Only stored here; the task
      // uses it when use_offset is on. A rejected message counts as not received.
      offset_subscriptions_[task->id()] =
        node->template create_subscription<geometry_msgs::msg::PoseStamped>(
        prefix + "/offset", rclcpp::QoS(1),
        [this, weak = std::weak_ptr<PoseTaskCommon>(pose_task), clock = node->get_clock(),
        logger = node->get_logger()](const geometry_msgs::msg::PoseStamped::SharedPtr msg) {
          auto task = weak.lock();
          if (!task || !msg || !active_.load()) {return;}
          const auto & p = msg->pose.position;
          const auto & o = msg->pose.orientation;
          Eigen::Quaterniond q(o.w, o.x, o.y, o.z);
          const std::string expected = task->offset_frame();
          if (!msg->header.frame_id.empty() && !expected.empty() && msg->header.frame_id != expected) {
            RCLCPP_WARN_THROTTLE(logger, *clock, 2000,
              "Offset for '%s' ignored: expressed in '%s', expected '%s'", task->id().c_str(),
              msg->header.frame_id.c_str(), expected.c_str());
            return;
          }
          if (!std::isfinite(p.x) || !std::isfinite(p.y) || !std::isfinite(p.z) ||
            !q.coeffs().allFinite() || q.norm() < 1e-6)
          {
            RCLCPP_WARN_THROTTLE(logger, *clock, 2000, "Offset for '%s' ignored: invalid pose",
              task->id().c_str());
            return;
          }
          Eigen::Isometry3d offset = Eigen::Isometry3d::Identity();
          offset.translation() = Eigen::Vector3d(p.x, p.y, p.z);
          offset.linear() = q.normalized().toRotationMatrix();
          task->set_offset(offset);
        });
      use_offset_publishers_[task->id()] = node->template create_publisher<std_msgs::msg::Bool>(
        prefix + "/use_offset", rclcpp::QoS(1).reliable().transient_local());
      subscriptions_[task->id()] =
        node->template create_subscription<geometry_msgs::msg::TwistStamped>(
        prefix + "/feedforward", rclcpp::QoS(1),
        [this, weak = std::weak_ptr<PoseTaskCommon>(pose_task),
        world_frame, base_frame, host_mutex, mutation_guard, logger = node->get_logger()](
          const geometry_msgs::msg::TwistStamped::SharedPtr msg) {
          std::unique_lock<std::mutex> callback_lock(callback_mutex_);
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
          // Never acquire the host mutex while holding callback_mutex_: the
          // lifecycle uses the opposite order when clearing all feedforward.
          callback_lock.unlock();
          if (host_mutex) {
            std::scoped_lock host_lock(*host_mutex);
            if (!active_.load() || task->feedforward_epoch() != epoch) {return;}
            std::string message;
            if (mutation_guard && !mutation_guard(task->id(), message)) {return;}
            // A deliberate release can enable the task and refresh its epoch.
            if (active_.load()) {task->set_feedforward(velocity, task->feedforward_epoch());}
          } else if (active_.load()) {task->set_feedforward(velocity, epoch);}
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
    // Latched state of each task's controlled point (true = offset point), published on change so
    // the target markers can draw a frame instead of the gripper.
    for (const auto & entry : use_offset_publishers_) {
      const auto task = tasks_.find(entry.first);
      const auto live = task == tasks_.end() ? nullptr : task->second.lock();
      if (!live) {continue;}
      const bool now = live->use_offset();
      const auto last = published_use_offset_.find(entry.first);
      if (last != published_use_offset_.end() && last->second == now) {continue;}
      published_use_offset_[entry.first] = now;
      std_msgs::msg::Bool state;
      state.data = now;
      entry.second->publish(state);
    }
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
  std::map<std::string, rclcpp::Subscription<geometry_msgs::msg::PoseStamped>::SharedPtr> offset_subscriptions_;
  std::map<std::string, rclcpp::Publisher<std_msgs::msg::Bool>::SharedPtr> use_offset_publishers_;
  std::map<std::string, bool> published_use_offset_;
};
}  // namespace task_priority_kinematic_control
