#pragma once

#include "sura_msgs/msg/navigator.hpp"

#include <chrono>

namespace task_priority_kinematic_control
{

// Transport the message and its receipt time as one realtime-buffer snapshot.
// Freshness is wall time since receipt; trajectory timing remains ROS time.
struct NavigationSample
{
  using Clock = std::chrono::steady_clock;
  sura_msgs::msg::Navigator::ConstSharedPtr message;
  Clock::time_point received{};

  bool is_fresh(double timeout, Clock::time_point now = Clock::now()) const
  {
    if (!message) {return false;}
    if (timeout <= 0.0) {return true;}
    const auto age = now - received;
    return age >= Clock::duration::zero() && std::chrono::duration<double>(age).count() <= timeout;
  }
};

}  // namespace task_priority_kinematic_control
