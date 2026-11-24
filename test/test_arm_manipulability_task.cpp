#include "task_priority_kinematic_control/tasks/arm_manipulability_task.hpp"
#include "task_priority_kinematic_control/core/capsule_observer.hpp"
#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"
#include "task_priority_kinematic_control/core/stpik_check.hpp"

#include <gtest/gtest.h>

#include <cmath>

namespace task_priority_kinematic_control
{
namespace
{

constexpr double kA2 = 0.2;
constexpr double kA3 = 0.2;

// Right arm: yaw (q1), shoulder pitch (q2), elbow pitch (q3), tool roll (q4, no position effect).
// Analytic w = a2 a3 |sin q3| |a2 cos q2 + a3 cos(q2 + q3)|: singular when stretched (q3 = 0).
Eigen::Vector3d tip_position(const Eigen::Vector4d & q)
{
  const double r = kA2 * std::cos(q(1)) + kA3 * std::cos(q(1) + q(2));
  const double z = kA2 * std::sin(q(1)) + kA3 * std::sin(q(1) + q(2));
  return {r * std::cos(q(0)), r * std::sin(q(0)), z};
}

class ArmBackend : public KinematicsBackend
{
public:
  explicit ArmBackend(std::shared_ptr<WholeBodyModel> model) : model_(std::move(model)) {}

  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState & state) override {current_ = compute_frame_state(state, "right_tip");}
  FrameState get_frame_state(const std::string &) const override {return current_;}
  FrameState compute_frame_state(const WholeBodyState & state, const std::string &) const override
  {
    ++evaluations;
    // right_offset() is a generalized-velocity column; joints exclude the base DoFs.
    const Eigen::Index column = static_cast<Eigen::Index>(model_->right_offset());
    const Eigen::Index offset = column - static_cast<Eigen::Index>(model_->base_dofs());
    const Eigen::Vector4d q = state.joint_positions.segment<4>(offset);
    FrameState frame;
    frame.pose.translation() = tip_position(q);
    frame.jacobian = Eigen::MatrixXd::Zero(6, static_cast<Eigen::Index>(model_->total_dofs()));
    frame.jacobian.block<3, 3>(0, 0).setIdentity();  // base translation
    const double h = 1e-7;
    for (Eigen::Index i = 0; i < 4; ++i) {
      Eigen::Vector4d qp = q, qm = q;
      qp(i) += h;
      qm(i) -= h;
      frame.jacobian.block<3, 1>(0, column + i) =
        (tip_position(qp) - tip_position(qm)) / (2.0 * h);
    }
    return frame;
  }
  Eigen::Isometry3d get_relative_transform(const std::string &, const std::string &) const override
  {return Eigen::Isometry3d::Identity();}
  const std::vector<CollisionCapsule> & collision_capsules() const override {return capsules_;}
  std::string name() const override {return "arm_test";}
  std::shared_ptr<KinematicsBackend> clone() const override
  {return std::make_shared<ArmBackend>(*this);}

  mutable int evaluations = 0;

private:
  std::shared_ptr<WholeBodyModel> model_;
  FrameState current_;
  std::vector<CollisionCapsule> capsules_;
};

double analytic_w(double q2, double q3)
{
  return kA2 * kA3 * std::abs(std::sin(q3)) * std::abs(kA2 * std::cos(q2) + kA3 * std::cos(q2 + q3));
}

std::map<std::string, rclcpp::Parameter> parameters()
{
  return {
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"priority", rclcpp::Parameter("priority", 24.0)},
    {"group", rclcpp::Parameter("group", "right")},
    {"arm", rclcpp::Parameter("arm", "right")},
    {"w_min", rclcpp::Parameter("w_min", 0.002)},
    {"w_des", rclcpp::Parameter("w_des", 0.004)},
    {"gain_scalar", rclcpp::Parameter("gain_scalar", 1.0)},
    {"max_velocity", rclcpp::Parameter("max_velocity", 0.3)}};
}

}  // namespace

class ArmManipulabilityTest : public ::testing::Test
{
protected:
  std::shared_ptr<WholeBodyModel> model = std::make_shared<WholeBodyModel>();
  std::unique_ptr<ArmBackend> backend;
  ArmManipulabilityTask task;
  WholeBodyState state;

  void SetUp() override
  {
    model->configure(
      "world", "base", "left_tip", "right_tip",
      {"l1", "l2", "l3", "l4"}, {"r1", "r2", "r3", "r4"}, {0, 1, 2, 3, 4, 5});
    backend = std::make_unique<ArmBackend>(model);
    TaskContext context;
    context.model = model;
    task.configure("right_manipulability", "task_priority_kinematic_control/ArmManipulabilityTask",
      parameters(), context);
    state.joint_positions = Eigen::VectorXd::Zero(8);
    state.joints_valid = true;
  }

  void set_right(double q1, double q2, double q3, double q4)
  {
    state.joint_positions.segment<4>(4) << q1, q2, q3, q4;
  }

  TaskComputation run()
  {
    backend->update(state);
    return task.update(state, *backend);
  }
};

TEST_F(ArmManipulabilityTest, ManipulabilityMatchesAnalyticValue)
{
  set_right(0.4, 0.3, 1.2, 0.7);
  backend->update(state);
  EXPECT_NEAR(task.manipulability(backend->get_frame_state("right_tip").jacobian),
    analytic_w(0.3, 1.2), 1e-8);
}

TEST_F(ArmManipulabilityTest, WellConditionedArmIsInactive)
{
  set_right(0.0, 0.3, 1.5, 0.0);
  const int before = backend->evaluations;
  const auto out = run();
  EXPECT_FALSE(out.active);
  EXPECT_TRUE(out.structural_zero_rows);
  EXPECT_EQ(out.jacobian.rows(), 1);
  EXPECT_TRUE(out.jacobian.isZero());
  EXPECT_EQ(out.status_message, "well_conditioned");
  // No gradient evaluations while inactive (only the cached update evaluation).
  EXPECT_EQ(backend->evaluations - before, 1);
}

TEST_F(ArmManipulabilityTest, NearSingularArmMovesAlongIncreasingManipulability)
{
  set_right(0.2, 0.3, 0.05, 0.1);
  const double w0 = analytic_w(0.3, 0.05);
  ASSERT_LT(w0, 0.002);
  const auto out = run();
  ASSERT_TRUE(out.active);
  EXPECT_EQ(out.status_message, "raising_manipulability");

  // Unit row restricted to the right-arm columns.
  EXPECT_NEAR(out.jacobian.row(0).norm(), 1.0, 1e-9);
  const Eigen::Index right = static_cast<Eigen::Index>(model->right_offset());
  EXPECT_TRUE(out.jacobian.leftCols(right).isZero());
  // Roll joint does not change the tip position, so it gets no motion.
  EXPECT_NEAR(out.jacobian(0, right + 3), 0.0, 1e-6);
  // Elbow dominates the escape direction from the stretched singularity.
  EXPECT_GT(std::abs(out.jacobian(0, right + 2)), 0.9);

  EXPECT_NEAR(out.error(0), (0.004 - w0) / 0.004, 1e-6);
  EXPECT_GT(out.desired_velocity(0), 0.0);
  EXPECT_LE(out.desired_velocity(0), 0.3);

  // Following the task row increases w.
  const double dt = 0.05;
  const Eigen::VectorXd qdot = out.jacobian.row(0).transpose() * out.desired_velocity(0);
  state.joint_positions += dt * qdot.tail(8);
  const double w1 = analytic_w(state.joint_positions(5), state.joint_positions(6));
  EXPECT_GT(w1, w0);
}

TEST_F(ArmManipulabilityTest, HysteresisKeepsRowUntilDesiredValue)
{
  set_right(0.0, 0.3, 0.2, 0.0);  // w ~= 0.0029, between w_min and w_des
  EXPECT_FALSE(run().active);
  set_right(0.0, 0.3, 0.05, 0.0);  // below w_min -> activates
  EXPECT_TRUE(run().active);
  set_right(0.0, 0.3, 0.2, 0.0);  // still below w_des -> stays active
  EXPECT_TRUE(run().active);
  set_right(0.0, 0.3, 1.5, 0.0);  // above w_des -> releases
  EXPECT_FALSE(run().active);
  set_right(0.0, 0.3, 0.2, 0.0);
  EXPECT_FALSE(run().active);
}

TEST_F(ArmManipulabilityTest, VelocityIsSaturated)
{
  std::string message;
  ASSERT_TRUE(task.set_gain({100.0}, message));
  set_right(0.0, 0.3, 0.05, 0.0);
  const auto out = run();
  ASSERT_TRUE(out.active);
  EXPECT_DOUBLE_EQ(out.desired_velocity(0), 0.3);
  EXPECT_FALSE(task.set_gain_scalar(-1.0, message));
}

TEST_F(ArmManipulabilityTest, DisabledTaskIsInactiveAndStatusReportsW)
{
  set_right(0.0, 0.3, 0.05, 0.0);
  ASSERT_TRUE(run().active);
  const auto status = task.build_status();
  ASSERT_EQ(status.error.size(), 3U);
  EXPECT_NEAR(status.error[0], analytic_w(0.3, 0.05), 1e-8);
  EXPECT_TRUE(status.active);

  task.set_enabled(false);
  const auto out = run();
  EXPECT_FALSE(out.active);
  EXPECT_EQ(out.status_message, "disabled");
}

TEST_F(ArmManipulabilityTest, EmptyAutoDeclaredFrameAndJointsUseArmDefaults)
{
  TaskContext context;
  context.model = model;
  auto params = parameters();
  params["frame_id"] = rclcpp::Parameter("frame_id", "");
  params["joint_names"] = rclcpp::Parameter("joint_names", std::vector<std::string>{});
  ArmManipulabilityTask other;
  ASSERT_NO_THROW(other.configure("x", "p", params, context));
  EXPECT_EQ(other.build_status().joint_names, model->right_arm_joints());
}

TEST_F(ArmManipulabilityTest, InvalidParametersThrow)
{
  TaskContext context;
  context.model = model;
  auto bad = parameters();
  bad["w_des"] = rclcpp::Parameter("w_des", 0.001);
  ArmManipulabilityTask other;
  EXPECT_THROW(other.configure("x", "p", bad, context), std::invalid_argument);

  auto no_arm = parameters();
  no_arm.erase("arm");
  EXPECT_THROW(other.configure("x", "p", no_arm, context), std::invalid_argument);

  auto bad_arm = parameters();
  bad_arm["arm"] = rclcpp::Parameter("arm", "middle");
  EXPECT_THROW(other.configure("x", "p", bad_arm, context), std::invalid_argument);
}

TEST_F(ArmManipulabilityTest, DebugSnapshotDrawsEllipsoidColouredByState)
{
  task.set_debug(true);
  set_right(0.0, 0.3, 0.05, 0.0);  // near singular -> active (red)
  ASSERT_TRUE(run().active);
  task.observe_command(WholeBodyCommand{}, 1000);
  const auto * snapshot = task.consume_snapshot();
  ASSERT_NE(snapshot, nullptr);
  ASSERT_TRUE(snapshot->valid);
  EXPECT_TRUE(snapshot->active);
  // Product of the singular values equals w.
  EXPECT_NEAR(snapshot->singular_values.prod(), snapshot->w, 1e-9);
  EXPECT_NEAR(snapshot->tip_position.x(), tip_position({0.0, 0.3, 0.05, 0.0}).x(), 1e-9);

  auto markers = arm_manipulability_markers(*snapshot, "world", "right_manipulability", 2.0);
  ASSERT_EQ(markers.markers.size(), 1U);
  auto & ellipsoid = markers.markers.front();
  EXPECT_EQ(ellipsoid.action, visualization_msgs::msg::Marker::ADD);
  EXPECT_EQ(ellipsoid.type, visualization_msgs::msg::Marker::SPHERE);
  EXPECT_FLOAT_EQ(ellipsoid.color.r, 1.0f);
  EXPECT_FLOAT_EQ(ellipsoid.color.g, 0.0f);
  // Flattened direction keeps a minimum visible thickness.
  const double thinnest = std::min({ellipsoid.scale.x, ellipsoid.scale.y, ellipsoid.scale.z});
  EXPECT_GE(thinnest, 0.003);
  const double widest = std::max({ellipsoid.scale.x, ellipsoid.scale.y, ellipsoid.scale.z});
  EXPECT_NEAR(widest, 2.0 * 0.5 * snapshot->singular_values.maxCoeff(), 1e-9);

  // Between w_min and w_des and inactive -> yellow; above w_des -> green.
  ArmManipulabilitySnapshot copy = *snapshot;
  copy.active = false;
  copy.w = 0.003;
  EXPECT_FLOAT_EQ(arm_manipulability_markers(copy, "world", "id", 2.0).markers[0].color.g, 0.9f);
  copy.w = 0.005;
  const auto green = arm_manipulability_markers(copy, "world", "id", 2.0).markers[0].color;
  EXPECT_FLOAT_EQ(green.r, 0.0f);
  EXPECT_FLOAT_EQ(green.g, 1.0f);

  EXPECT_EQ(delete_arm_manipulability_markers("world", "id").markers[0].action,
    visualization_msgs::msg::Marker::DELETE);
}

TEST_F(ArmManipulabilityTest, NoSnapshotWithoutDebug)
{
  set_right(0.0, 0.3, 0.05, 0.0);
  run();
  task.observe_command(WholeBodyCommand{}, 1000);
  EXPECT_EQ(task.consume_snapshot(), nullptr);
}

TEST_F(ArmManipulabilityTest, StpikCanIsolateActiveTask)
{
  std::shared_ptr<KinematicsBackend> shared_backend = std::make_shared<ArmBackend>(model);
  auto shared_task = std::make_shared<ArmManipulabilityTask>(task);
  std::vector<std::shared_ptr<TaskBase>> tasks{shared_task};
  HierarchySolver solver;
  const auto snapshot = StpikTaskAccess::capture(state, model, shared_backend, solver, tasks);
  EXPECT_TRUE(snapshot.unavailable_reason.empty()) << snapshot.unavailable_reason;
  ASSERT_EQ(snapshot.tasks.size(), 1U);
  EXPECT_NE(dynamic_cast<ArmManipulabilityTask *>(snapshot.tasks.front().get()), nullptr);
}

}  // namespace task_priority_kinematic_control
