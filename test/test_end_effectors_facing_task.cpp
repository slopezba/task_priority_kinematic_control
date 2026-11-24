#include "task_priority_kinematic_control/tasks/end_effectors_facing_task.hpp"

#include <gtest/gtest.h>

#include <cmath>
#include <limits>

namespace task_priority_kinematic_control
{
namespace
{

constexpr double kPi = 3.14159265358979323846;

// Each EE is a free-floating body: 6 "joints" per EE (world translation, world rotation).
class FreeBodiesBackend : public KinematicsBackend
{
public:
  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState &) override {}

  FrameState get_frame_state(const std::string & frame_id) const override
  {
    FrameState out;
    out.jacobian = Eigen::MatrixXd::Zero(6, 12);
    const Eigen::Index offset = frame_id == "left_tip" ? 0 : 6;
    out.pose = frame_id == "left_tip" ? left : right;
    out.jacobian.block<3, 3>(0, offset) = Eigen::Matrix3d::Identity();
    out.jacobian.block<3, 3>(3, offset + 3) = Eigen::Matrix3d::Identity();
    return out;
  }

  Eigen::Isometry3d get_relative_transform(
    const std::string & from_frame, const std::string & to_frame) const override
  {
    return get_frame_state(from_frame).pose.inverse() * get_frame_state(to_frame).pose;
  }

  const std::vector<CollisionCapsule> & collision_capsules() const override {return capsules_;}
  std::string name() const override {return "free_bodies";}

  // Applies a world-frame twist step [v_left, w_left, v_right, w_right] * dt.
  void apply(const Eigen::VectorXd & qdot, double dt)
  {
    apply_to(left, qdot.segment<6>(0) * dt);
    apply_to(right, qdot.segment<6>(6) * dt);
  }

  Eigen::Isometry3d left = Eigen::Isometry3d::Identity();
  Eigen::Isometry3d right = Eigen::Isometry3d::Identity();

private:
  static void apply_to(Eigen::Isometry3d & pose, const Eigen::Matrix<double, 6, 1> & step)
  {
    pose.translation() += step.head<3>();
    const Eigen::Vector3d w = step.tail<3>();
    if (w.norm() > 0.0) {
      pose.linear() = Eigen::AngleAxisd(w.norm(), w.normalized()).toRotationMatrix() * pose.linear();
    }
  }

  std::vector<CollisionCapsule> capsules_;
};

std::shared_ptr<WholeBodyModel> model()
{
  auto out = std::make_shared<WholeBodyModel>();
  out->configure("world", "base", "left_tip", "right_tip", {"left_joint"}, {"right_joint"}, {});
  return out;
}

std::map<std::string, rclcpp::Parameter> parameters(
  const std::vector<double> & extra_roll = {}, const std::vector<double> & extra_distance = {})
{
  std::map<std::string, rclcpp::Parameter> out = {
    {"plugin", rclcpp::Parameter("plugin", "task_priority_kinematic_control/EndEffectorsFacingTask")},
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 1.0)},
    {"group", rclcpp::Parameter("group", "bimanual")},
    {"left_frame", rclcpp::Parameter("left_frame", "left_tip")},
    {"right_frame", rclcpp::Parameter("right_frame", "right_tip")},
    {"left_pointing_axis", rclcpp::Parameter("left_pointing_axis", std::vector<double>{0.0, 0.0, 1.0})},
    {"right_pointing_axis", rclcpp::Parameter("right_pointing_axis", std::vector<double>{0.0, 0.0, 1.0})},
    {"gain", rclcpp::Parameter("gain", std::vector<double>{1.0, 1.0, 1.0, 1.0, 1.0, 1.0})},
  };
  if (!extra_roll.empty()) {
    out.emplace("roll_angle", rclcpp::Parameter("roll_angle", extra_roll.front()));
  }
  if (!extra_distance.empty()) {
    out.emplace("distance", rclcpp::Parameter("distance", extra_distance.front()));
  }
  return out;
}

EndEffectorsFacingTask make_task(
  const std::vector<double> & roll = {0.0}, const std::vector<double> & distance = {0.25})
{
  EndEffectorsFacingTask task;
  task.configure(
    "EndEffectors_Facing", "task_priority_kinematic_control/EndEffectorsFacingTask",
    parameters(roll, distance), TaskContext{model()});
  return task;
}

msg::FacingTarget target_msg(
  const Eigen::Vector3d & left_axis, const Eigen::Vector3d & right_axis,
  double roll, double distance)
{
  msg::FacingTarget out;
  out.left_pointing_axis.x = left_axis.x();
  out.left_pointing_axis.y = left_axis.y();
  out.left_pointing_axis.z = left_axis.z();
  out.right_pointing_axis.x = right_axis.x();
  out.right_pointing_axis.y = right_axis.y();
  out.right_pointing_axis.z = right_axis.z();
  out.roll_angle = roll;
  out.distance = distance;
  return out;
}

// Left at origin (identity), right on +z at `distance` with its z pointing back (-z).
FreeBodiesBackend facing_backend(double distance, double right_rotation_about_line = 0.0)
{
  FreeBodiesBackend backend;
  backend.right.translation() = Eigen::Vector3d(0.0, 0.0, distance);
  // Flip about x: z -> -z, x -> x (roll 0 with the cyclic convention z -> x).
  const Eigen::Matrix3d flip = Eigen::AngleAxisd(kPi, Eigen::Vector3d::UnitX()).toRotationMatrix();
  backend.right.linear() =
    Eigen::AngleAxisd(right_rotation_about_line, Eigen::Vector3d::UnitZ()).toRotationMatrix() * flip;
  return backend;
}

Eigen::Matrix3d rotation_from_vector(const Eigen::Vector3d & v)
{
  return Eigen::AngleAxisd(v.norm(), v.normalized()).toRotationMatrix();
}

// Finite-difference check of the current-rate jacobian: J = -d(error)/dq for a fixed target.
void expect_jacobian_matches(
  EndEffectorsFacingTask task, const FreeBodiesBackend & backend,
  const std::vector<Eigen::Index> & rows, double tolerance)
{
  WholeBodyState state;
  const auto nominal = task.update(state, backend);
  ASSERT_TRUE(nominal.active);
  ASSERT_EQ(nominal.jacobian.rows(), 6);
  const double h = 1e-6;
  for (Eigen::Index col = 0; col < 12; ++col) {
    FreeBodiesBackend plus = backend;
    FreeBodiesBackend minus = backend;
    Eigen::VectorXd step = Eigen::VectorXd::Zero(12);
    step(col) = 1.0;
    plus.apply(step, h);
    minus.apply(step, -h);
    const auto out_plus = task.update(state, plus);
    const auto out_minus = task.update(state, minus);
    for (const auto row : rows) {
      const double numeric = -(out_plus.error(row) - out_minus.error(row)) / (2.0 * h);
      EXPECT_NEAR(nominal.jacobian(row, col), numeric, tolerance)
        << "row " << row << " col " << col;
    }
  }
}

}  // namespace

TEST(EndEffectorsFacingTask, ReportsFacingTargetType)
{
  auto task = make_task();
  const auto status = task.build_status();
  EXPECT_EQ(status.id, "EndEffectors_Facing");
  EXPECT_EQ(status.target_type, "facing");
  const auto target = task.current_target();
  ASSERT_EQ(target.size(), 8u);
  EXPECT_DOUBLE_EQ(target[2], 1.0);
  EXPECT_DOUBLE_EQ(target[5], 1.0);
  EXPECT_DOUBLE_EQ(target[7], 0.25);
}

TEST(EndEffectorsFacingTask, ZeroErrorWhenFacing)
{
  auto task = make_task({0.0}, {0.25});
  WholeBodyState state;
  const auto out = task.update(state, facing_backend(0.25));
  ASSERT_TRUE(out.active);
  ASSERT_EQ(out.error.size(), 6);
  EXPECT_LT(out.error.norm(), 1e-9);
}

TEST(EndEffectorsFacingTask, RollConventionZAxes)
{
  auto task = make_task();
  // x axes parallel -> roll 0.
  auto measured = task.measure(facing_backend(0.25));
  ASSERT_TRUE(measured.roll_valid);
  EXPECT_NEAR(measured.roll, 0.0, 1e-9);
  // x axes opposite (= y axes parallel) -> roll pi.
  measured = task.measure(facing_backend(0.25, kPi));
  ASSERT_TRUE(measured.roll_valid);
  EXPECT_NEAR(std::abs(measured.roll), kPi, 1e-9);
  // Right EE turned about the line: roll follows with sign around d (left -> right = +z).
  measured = task.measure(facing_backend(0.25, 0.4));
  EXPECT_NEAR(measured.roll, 0.4, 1e-9);
}

TEST(EndEffectorsFacingTask, RollWrapsAroundPi)
{
  auto task = make_task({kPi - 0.05}, {0.25});
  WholeBodyState state;
  const auto out = task.update(state, facing_backend(0.25, -kPi + 0.05));
  ASSERT_TRUE(out.active);
  EXPECT_NEAR(out.error(4), -0.1, 1e-9);
}

TEST(EndEffectorsFacingTask, JacobianMatchesFiniteDifferencesAtFacingConfiguration)
{
  auto task = make_task({0.3}, {0.4});
  expect_jacobian_matches(task, facing_backend(0.25, 0.1), {0, 1, 2, 3, 4, 5}, 1e-5);
}

TEST(EndEffectorsFacingTask, RollAndDistanceJacobianExactAtGenericConfiguration)
{
  auto task = make_task({0.3}, {0.4});
  FreeBodiesBackend backend;
  backend.left.linear() = rotation_from_vector(Eigen::Vector3d(0.2, -0.3, 0.1));
  backend.right.translation() = Eigen::Vector3d(0.1, 0.05, 0.3);
  backend.right.linear() = rotation_from_vector(Eigen::Vector3d(2.5, 0.4, -0.3));
  expect_jacobian_matches(task, backend, {4, 5}, 1e-5);
}

TEST(EndEffectorsFacingTask, ConvergesFromGenericConfiguration)
{
  auto task = make_task({0.0}, {0.25});
  std::string message;
  ASSERT_TRUE(task.set_facing_target(
    target_msg(Eigen::Vector3d::UnitZ(), -Eigen::Vector3d::UnitY(), 0.5, 0.3), message)) << message;

  FreeBodiesBackend backend;
  backend.left.linear() = rotation_from_vector(Eigen::Vector3d(0.3, -0.4, 0.2));
  backend.right.translation() = Eigen::Vector3d(0.15, 0.05, 0.35);
  backend.right.linear() = rotation_from_vector(Eigen::Vector3d(1.2, 0.6, -0.4));

  WholeBodyState state;
  const double dt = 0.01;
  TaskComputation out;
  for (int i = 0; i < 3000; ++i) {
    out = task.update(state, backend);
    ASSERT_TRUE(out.active);
    const Eigen::VectorXd qdot =
      out.jacobian.completeOrthogonalDecomposition().solve(out.desired_velocity);
    backend.apply(qdot, dt);
  }
  out = task.update(state, backend);
  EXPECT_LT(out.error.norm(), 1e-4);

  const Eigen::Vector3d delta = backend.right.translation() - backend.left.translation();
  const Eigen::Vector3d d = delta.normalized();
  EXPECT_NEAR(delta.norm(), 0.3, 1e-4);
  EXPECT_NEAR(backend.left.linear().col(2).dot(d), 1.0, 1e-6);    // left +z towards right
  EXPECT_NEAR(-backend.right.linear().col(1).dot(-d), 1.0, 1e-6); // right -y towards left
  const auto measured = task.measure(backend);
  EXPECT_NEAR(measured.roll, 0.5, 1e-4);
}

TEST(EndEffectorsFacingTask, NanRollKeepsCurrentRoll)
{
  auto task = make_task({0.0}, {0.25});
  std::string message;
  ASSERT_TRUE(task.set_facing_target(
    target_msg(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ(),
    std::numeric_limits<double>::quiet_NaN(), 0.25), message)) << message;
  WholeBodyState state;
  const auto out = task.update(state, facing_backend(0.25, 0.7));
  ASSERT_TRUE(out.active);
  EXPECT_NEAR(out.error(4), 0.0, 1e-12);
  EXPECT_NEAR(task.current_target()[6], 0.7, 1e-9);
}

TEST(EndEffectorsFacingTask, MissingDistanceAndRollAreCaptured)
{
  EndEffectorsFacingTask task;
  task.configure(
    "EndEffectors_Facing", "task_priority_kinematic_control/EndEffectorsFacingTask",
    parameters(), TaskContext{model()});
  WholeBodyState state;
  const auto out = task.update(state, facing_backend(0.33, 0.2));
  ASSERT_TRUE(out.active);
  EXPECT_LT(out.error.norm(), 1e-9);
  EXPECT_NEAR(task.current_target()[6], 0.2, 1e-9);
  EXPECT_NEAR(task.current_target()[7], 0.33, 1e-9);
}

TEST(EndEffectorsFacingTask, RejectsInvalidTargets)
{
  auto task = make_task();
  std::string message;
  EXPECT_FALSE(task.set_facing_target(
    target_msg(Eigen::Vector3d(0.71, 0.21, 0.1), Eigen::Vector3d::UnitZ(), 0.0, 0.25), message));
  EXPECT_FALSE(task.set_facing_target(
    target_msg(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::Zero(), 0.0, 0.25), message));
  EXPECT_FALSE(task.set_facing_target(
    target_msg(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ(), 0.0, 0.0), message));
  EXPECT_FALSE(task.set_facing_target(
    target_msg(Eigen::Vector3d::UnitZ(), Eigen::Vector3d::UnitZ(),
    std::numeric_limits<double>::infinity(), 0.25), message));
  EXPECT_TRUE(task.set_facing_target(
    target_msg(-Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(), 0.0, 0.25), message)) << message;
}

TEST(EndEffectorsFacingTask, InactiveWhenTooClose)
{
  auto task = make_task();
  WholeBodyState state;
  const auto out = task.update(state, facing_backend(0.01));
  EXPECT_FALSE(out.active);
  EXPECT_EQ(out.status_message, "ee too close");
}

TEST(EndEffectorsFacingTask, ActivationSelectsRows)
{
  auto params = parameters({0.0}, {0.25});
  params["activation"] = rclcpp::Parameter("activation", std::vector<int64_t>{1, 1, 1, 1, 0, 1});
  EndEffectorsFacingTask task;
  task.configure(
    "EndEffectors_Facing", "task_priority_kinematic_control/EndEffectorsFacingTask",
    params, TaskContext{model()});
  WholeBodyState state;
  const auto out = task.update(state, facing_backend(0.3));
  ASSERT_TRUE(out.active);
  EXPECT_EQ(out.error.size(), 5);
  EXPECT_NEAR(out.error(4), -0.05, 1e-9);
}

}  // namespace task_priority_kinematic_control
