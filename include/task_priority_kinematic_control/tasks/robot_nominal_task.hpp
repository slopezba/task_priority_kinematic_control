#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"
#include "sura_manipulation_actions/msg/robot_nominal_reference.hpp"
#include <chrono>

namespace task_priority_kinematic_control
{

// Low-priority whole-body nominal: base pose (world, ZYX coordinates) and every arm
// joint. References are streamed (pose + velocity feedforward) and expire after
// joint_command_timeout; without a live reference the task contributes no rows.
// Rows: [x, y, z, roll, pitch, yaw, all model joints] filtered by `activation`.
class RobotNominalTask : public TaskBaseCommon
{
  friend class StpikTaskAccess;
public:
  using Coordinates = Eigen::Matrix<double, 6, 1>;
  void configure(
    const std::string & id, const std::string & plugin_name,
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const TaskContext & context) override;
  TaskComputation update(const WholeBodyState & state, const KinematicsBackend & backend) override;

  // base: ZYX coordinates; base_rates: their time derivatives; joints in model order.
  // mask: empty, or one flag per row (see RobotNominalReference.msg).
  bool set_reference(const Coordinates & base, const Coordinates & base_rates,
    const Eigen::VectorXd & joints, const Eigen::VectorXd & joint_velocities, std::string & message,
    const std::vector<bool> & mask = {});
  // True when the current reference commands no base row at all.
  bool base_masked() const;
  bool set_reference(const sura_manipulation_actions::msg::RobotNominalReference & reference,
    std::string & message, bool zero_velocity = false);
  void clear_reference();
  bool has_live_reference() const;

  bool set_gain(const std::vector<double> & gain, std::string & message) override;
  bool set_joint_activation(const std::vector<bool> & activation, std::string & message) override;
  bool set_enabled(bool enabled) override;
  void reset() override;
  msg::TaskStatus build_status() const override;
  std::vector<double> current_target() const override;

  // Hold tasks retargeted with every reference (base FramePoseTask, JointNominalTask).
  const std::vector<std::string> & sync_tasks() const {return sync_tasks_;}
  size_t rows() const {return 6 + joint_names_.size();}
  const WholeBodyModel & model() const {return *context_.model;}
  const Coordinates & base_target() const {return base_target_;}
  const Eigen::VectorXd & joint_target() const {return joint_target_;}

private:
  bool live() const;
  std::vector<std::string> joint_names_;
  std::vector<std::string> sync_tasks_;
  Eigen::VectorXd gains_;
  std::vector<bool> activation_;
  std::vector<bool> reference_mask_;
  Coordinates base_target_ = Coordinates::Zero();
  Coordinates base_rates_ = Coordinates::Zero();
  Eigen::VectorXd joint_target_;
  Eigen::VectorXd joint_velocity_;
  bool has_reference_ = false;
  double timeout_ = 0.5;
  std::chrono::steady_clock::time_point reference_time_;
  std::string message_ = "idle";
};

// Apply one reference to robot_nominal and copy it into its configured hold tasks,
// so that holds (base_pose, *_arm_nominal) neither fight the nominal nor pull the
// robot back when the stream ends. Shared by the runtime IO and offline simulation.
bool apply_robot_nominal_reference(
  const std::vector<std::shared_ptr<TaskBase>> & tasks, RobotNominalTask & task,
  const RobotNominalTask::Coordinates & base, const RobotNominalTask::Coordinates & base_rates,
  const Eigen::VectorXd & joints, const Eigen::VectorXd & joint_velocities, std::string & message,
  const std::vector<bool> & mask = {});
bool apply_robot_nominal_reference(
  const std::vector<std::shared_ptr<TaskBase>> & tasks, RobotNominalTask & task,
  const sura_manipulation_actions::msg::RobotNominalReference & reference, std::string & message,
  bool zero_velocity = false);
std::shared_ptr<RobotNominalTask> find_robot_nominal_task(
  const std::vector<std::shared_ptr<TaskBase>> & tasks);

}  // namespace task_priority_kinematic_control
