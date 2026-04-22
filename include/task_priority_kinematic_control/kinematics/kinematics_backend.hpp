#pragma once

#include "task_priority_kinematic_control/core/common.hpp"
#include "task_priority_kinematic_control/core/whole_body_model.hpp"

#include <rclcpp/logger.hpp>

#include <memory>
#include <string>

namespace task_priority_kinematic_control
{

class KinematicsBackend
{
public:
  virtual ~KinematicsBackend() = default;

  virtual void configure(
    const WholeBodyModel & model,
    const std::string & robot_description,
    const rclcpp::Logger & logger) = 0;

  virtual void update(const WholeBodyState & state) = 0;

  virtual FrameState get_frame_state(const std::string & frame_id) const = 0;

  virtual Eigen::Isometry3d get_relative_transform(
    const std::string & from_frame,
    const std::string & to_frame) const = 0;

  virtual std::string name() const = 0;
};

using KinematicsBackendPtr = std::shared_ptr<KinematicsBackend>;

}  // namespace task_priority_kinematic_control
