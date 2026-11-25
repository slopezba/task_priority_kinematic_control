// Offset del punto controlado (p. ej. la punta de la brida respecto al tool): pose y Jacobiano desplazados,
// use_offset en caliente, recaptura del objetivo, caducidad y topic de entrada.
#include "task_priority_kinematic_control/core/pose_task_io.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_position_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"

#include <gtest/gtest.h>
#include <chrono>
#include <limits>
#include <map>
#include <vector>
#include <cmath>
#include <string>
#include <thread>

namespace task_priority_kinematic_control
{
namespace
{
using std::chrono::milliseconds;
using std::chrono::seconds;

class OffsetBackend : public KinematicsBackend
{
public:
  FrameState frame;
  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState &) override {}
  FrameState get_frame_state(const std::string &) const override {return frame;}
  FrameState compute_frame_state(const WholeBodyState &, const std::string &) const override {return frame;}
  Eigen::Isometry3d get_relative_transform(const std::string &, const std::string &) const override
  {return Eigen::Isometry3d::Identity();}
  const std::vector<CollisionCapsule> & collision_capsules() const override {return capsules;}
  std::string name() const override {return "offset_test";}
  std::vector<CollisionCapsule> capsules;
};

Eigen::Isometry3d iso(const Eigen::Vector3d & translation, double yaw = 0.0)
{
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  t.translation() = translation;
  t.linear() = Eigen::AngleAxisd(yaw, Eigen::Vector3d::UnitZ()).toRotationMatrix();
  return t;
}

std::shared_ptr<WholeBodyModel> make_model()
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left", "right", {}, {}, {0, 1, 2, 3, 4, 5});
  return model;
}

std::map<std::string, rclcpp::Parameter> frame_parameters(const std::string & frame)
{
  return {{"frame_id", rclcpp::Parameter("frame_id", frame)}};
}

OffsetBackend identity_backend()
{
  OffsetBackend backend;
  backend.frame.pose = iso(Eigen::Vector3d(1.0, 0.0, 0.0));
  backend.frame.jacobian = Eigen::MatrixXd::Identity(6, 6);
  return backend;
}
}  // namespace

TEST(PoseOffsetMathTest, ShiftsPoseAndOnlyTheLinearJacobianRows)
{
  FrameState frame;
  frame.pose = iso(Eigen::Vector3d(1, 2, 3), M_PI / 2);
  frame.jacobian = Eigen::MatrixXd::Zero(6, 3);
  frame.jacobian << 1, 0, 2,
    0, 1, 0,
    0, 0, 3,
    0, 0, 1,
    1, 0, 0,
    0, 1, 0;
  const FrameState original = frame;
  const Eigen::Isometry3d offset = iso(Eigen::Vector3d(0.1, 0.05, -0.2), 0.3);
  pose_math::apply_tool_offset(frame, offset);

  EXPECT_TRUE(frame.pose.isApprox(original.pose * offset));
  EXPECT_TRUE(frame.jacobian.bottomRows(3).isApprox(original.jacobian.bottomRows(3)));
  const Eigen::Vector3d p = original.pose.rotation() * offset.translation();
  for (Eigen::Index c = 0; c < 3; ++c) {
    const Eigen::Vector3d expected = original.jacobian.col(c).head<3>() +
      original.jacobian.col(c).tail<3>().cross(p);
    // v_point = v_frame + omega x p; the Jacobian column maps a joint rate to that velocity.
    EXPECT_TRUE(frame.jacobian.col(c).head<3>().isApprox(expected)) << "column " << c;
  }
}

TEST(PoseOffsetMathTest, ZeroOffsetChangesNothing)
{
  FrameState frame;
  frame.pose = iso(Eigen::Vector3d(1, 2, 3), 0.4);
  frame.jacobian = Eigen::MatrixXd::Random(6, 5);
  const FrameState original = frame;
  pose_math::apply_tool_offset(frame, Eigen::Isometry3d::Identity());
  EXPECT_TRUE(frame.pose.isApprox(original.pose));
  EXPECT_TRUE(frame.jacobian.isApprox(original.jacobian));
}

TEST(PoseOffsetTaskTest, DisabledByDefaultAndRejectedWithoutAnOffset)
{
  FramePoseTask task;
  task.configure("pose", "plugin", frame_parameters("left"), TaskContext{make_model()});
  auto backend = identity_backend();
  EXPECT_FALSE(task.use_offset());
  EXPECT_FALSE(task.offset_ready());
  std::string message;
  EXPECT_FALSE(task.set_use_offset(true, message));
  EXPECT_FALSE(task.use_offset());
  EXPECT_FALSE(message.empty());
  // Without use_offset the task behaves exactly as before, even with an offset available.
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  const auto result = task.update(WholeBodyState{}, backend);
  ASSERT_TRUE(result.active);
  EXPECT_TRUE(result.frame_pose.isApprox(backend.frame.pose));
}

TEST(PoseOffsetTaskTest, ControlsTheOffsetPointAndRecapturesTheTarget)
{
  FramePoseTask task;
  task.configure("pose", "plugin", frame_parameters("left"), TaskContext{make_model()});
  auto backend = identity_backend();
  ASSERT_TRUE(task.update(WholeBodyState{}, backend).active);
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  ASSERT_TRUE(task.offset_ready());
  std::string message;
  ASSERT_TRUE(task.set_use_offset(true, message)) << message;

  const auto shifted = task.update(WholeBodyState{}, backend);
  ASSERT_TRUE(shifted.active);
  EXPECT_NEAR(shifted.frame_pose.translation().z(), 0.2, 1e-12);
  ASSERT_TRUE(shifted.has_updated_target);
  EXPECT_NEAR(shifted.updated_target.position.z, 0.2, 1e-12);
  // No jump: the target was captured at the offset point, so the error is zero.
  EXPECT_NEAR(shifted.error.norm(), 0.0, 1e-12);
  // With R = I the linear rows gain -skew(p) in the angular columns: J_lin = I - skew(p).
  Eigen::Matrix3d skew;
  skew << 0, -0.2, 0, 0.2, 0, 0, 0, 0, 0;
  const Eigen::Matrix3d linear_angular = shifted.jacobian.block<3, 3>(0, 3);
  EXPECT_TRUE(linear_angular.isApprox(-skew));

  ASSERT_TRUE(task.set_use_offset(false, message));
  const auto plain = task.update(WholeBodyState{}, backend);
  EXPECT_NEAR(plain.frame_pose.translation().z(), 0.0, 1e-12);
  EXPECT_NEAR(plain.error.norm(), 0.0, 1e-12);
}

TEST(PoseOffsetTaskTest, RecapturesEvenWithAutomaticTargetUpdateOff)
{
  FramePoseTask task;
  auto parameters = frame_parameters("left");
  parameters["auto_update_target_pose"] = rclcpp::Parameter("auto_update_target_pose", false);
  task.configure("pose", "plugin", parameters, TaskContext{make_model()});
  auto backend = identity_backend();
  geometry_msgs::msg::PoseStamped goal;
  goal.header.frame_id = "world";
  goal.pose.position.x = 5.0;
  goal.pose.orientation.w = 1.0;
  ASSERT_TRUE(task.set_pose_goal(goal));
  task.update(WholeBodyState{}, backend);
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  std::string message;
  ASSERT_TRUE(task.set_use_offset(true, message)) << message;
  const auto result = task.update(WholeBodyState{}, backend);
  EXPECT_NEAR(result.error.norm(), 0.0, 1e-12);
  const auto target = task.current_target();
  ASSERT_EQ(target.size(), 7U);
  EXPECT_NEAR(target[0], 1.0, 1e-12);
  EXPECT_NEAR(target[2], 0.2, 1e-12);
  // Later changes of the offset do not recapture: the target stays in the world and the error shows.
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.3))));
  const auto moved = task.update(WholeBodyState{}, backend);
  EXPECT_NEAR(moved.error(2), -0.1, 1e-12);
}

TEST(PoseOffsetTaskTest, RuntimeParameterGoesThroughTheFeedforwardOptionPath)
{
  FramePoseTask task;
  task.configure("pose", "plugin", frame_parameters("left"), TaskContext{make_model()});
  std::string message;
  EXPECT_FALSE(task.set_feedforward_option("use_offset", rclcpp::Parameter("use_offset", true), message));
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  EXPECT_TRUE(task.set_feedforward_option("use_offset", rclcpp::Parameter("use_offset", true), message));
  EXPECT_TRUE(task.use_offset());
  EXPECT_TRUE(task.set_feedforward_option("offset_timeout", rclcpp::Parameter("offset_timeout", 1.0), message));
  EXPECT_FALSE(task.set_feedforward_option("offset_timeout", rclcpp::Parameter("offset_timeout", -1.0), message));
  EXPECT_FALSE(task.set_feedforward_option("use_offset", rclcpp::Parameter("use_offset", 3.0), message));
}

TEST(PoseOffsetTaskTest, TheBaseFrameCannotUseAnOffset)
{
  FramePoseTask task;
  task.configure("pose", "plugin", frame_parameters("base"), TaskContext{make_model()});
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  std::string message;
  EXPECT_FALSE(task.set_use_offset(true, message));
  FramePoseTask configured;
  auto parameters = frame_parameters("base");
  parameters["use_offset"] = rclcpp::Parameter("use_offset", true);
  EXPECT_THROW(configured.configure("pose", "plugin", parameters, TaskContext{make_model()}),
    std::invalid_argument);
}

TEST(PoseOffsetTaskTest, StaleOffsetStopsTheTaskInsteadOfFallingBack)
{
  auto now = std::chrono::steady_clock::time_point{} + seconds(100);
  TaskContext context{make_model()};
  context.steady_now = [&now]() {return now;};
  FramePoseTask task;
  auto parameters = frame_parameters("left");
  parameters["offset_timeout"] = rclcpp::Parameter("offset_timeout", 0.5);
  task.configure("pose", "plugin", parameters, context);
  auto backend = identity_backend();
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  std::string message;
  ASSERT_TRUE(task.set_use_offset(true, message));
  now += milliseconds(400);
  EXPECT_TRUE(task.update(WholeBodyState{}, backend).active);
  now += milliseconds(200);
  const auto stale = task.update(WholeBodyState{}, backend);
  EXPECT_FALSE(stale.active);
  EXPECT_EQ(stale.status_message, "offset stale");
  // A fresh message resumes it.
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  EXPECT_TRUE(task.update(WholeBodyState{}, backend).active);
}

TEST(PoseOffsetTaskTest, EnabledFromTheConfigurationWaitsForTheFirstOffset)
{
  FramePoseTask task;
  auto parameters = frame_parameters("left");
  parameters["use_offset"] = rclcpp::Parameter("use_offset", true);
  task.configure("pose", "plugin", parameters, TaskContext{make_model()});
  auto backend = identity_backend();
  const auto waiting = task.update(WholeBodyState{}, backend);
  EXPECT_FALSE(waiting.active);
  EXPECT_EQ(waiting.status_message, "waiting for offset");
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  EXPECT_TRUE(task.update(WholeBodyState{}, backend).active);
}

TEST(PoseOffsetTaskTest, CopiesKeepTheOffsetState)
{
  FramePoseTask task;
  task.configure("pose", "plugin", frame_parameters("left"), TaskContext{make_model()});
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  std::string message;
  ASSERT_TRUE(task.set_use_offset(true, message));
  FramePoseTask copy(task);
  auto backend = identity_backend();
  EXPECT_TRUE(copy.use_offset());
  const auto result = copy.update(WholeBodyState{}, backend);
  ASSERT_TRUE(result.active);
  EXPECT_NEAR(result.frame_pose.translation().z(), 0.2, 1e-12);
}

TEST(PoseOffsetTaskTest, CopiesFreezeTheOffsetSoOfflineProbesDoNotExpireIt)
{
  auto now = std::chrono::steady_clock::time_point{} + seconds(100);
  TaskContext context{make_model()};
  context.steady_now = [&now]() {return now;};
  FramePoseTask task;
  auto parameters = frame_parameters("left");
  parameters["offset_timeout"] = rclcpp::Parameter("offset_timeout", 0.5);
  task.configure("pose", "plugin", parameters, context);
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  std::string message;
  ASSERT_TRUE(task.set_use_offset(true, message));
  FramePoseTask probe(task);  // as an STPIK snapshot does while the live offset is still fresh
  now += seconds(30);         // the probe runs for far longer than offset_timeout
  auto backend = identity_backend();
  EXPECT_FALSE(task.update(WholeBodyState{}, backend).active);
  const auto frozen = probe.update(WholeBodyState{}, backend);
  ASSERT_TRUE(frozen.active);
  EXPECT_NEAR(frozen.frame_pose.translation().z(), 0.2, 1e-12);
  // A copy of a task whose offset had already expired stays unavailable instead of using stale data.
  FramePoseTask late(task);
  EXPECT_FALSE(late.update(WholeBodyState{}, backend).active);
}

TEST(PoseOffsetTaskTest, ActiveOffsetAndTipStateFollowUseOffset)
{
  FramePoseTask task;
  task.configure("pose", "plugin", frame_parameters("left"), TaskContext{make_model()});
  auto backend = identity_backend();
  const WholeBodyState state;
  EXPECT_FALSE(task.active_offset().has_value());
  ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
  EXPECT_FALSE(task.active_offset().has_value());  // received, but use_offset is off
  EXPECT_NEAR(pose_math::compute_tip_state(backend, state, "left", &task).pose.translation().z(), 0.0, 1e-12);
  std::string message;
  ASSERT_TRUE(task.set_use_offset(true, message));
  ASSERT_TRUE(task.active_offset().has_value());
  const auto tip = pose_math::compute_tip_state(backend, state, "left", &task);
  EXPECT_NEAR(tip.pose.translation().z(), 0.2, 1e-12);
  EXPECT_NEAR(tip.jacobian(0, 4), 0.2, 1e-12);  // J_lin = I - skew(p): x row picks up omega_y * p_z
  // A task that is not a pose task, or no task, gives the plain frame state.
  EXPECT_NEAR(pose_math::compute_tip_state(backend, state, "left", nullptr).pose.translation().z(), 0.0, 1e-12);
}

TEST(PoseOffsetTaskTest, EndEffectorTasksShiftTheirFrameToo)
{
  auto backend = identity_backend();
  std::string message;
  {
    EndEffectorPoseTask task;
    task.configure("pose", "plugin", frame_parameters("left"), TaskContext{make_model()});
    ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
    ASSERT_TRUE(task.set_use_offset(true, message));
    const auto result = task.update(WholeBodyState{}, backend);
    ASSERT_TRUE(result.active);
    EXPECT_NEAR(result.frame_pose.translation().z(), 0.2, 1e-12);
  }
  {
    EndEffectorPositionTask task;
    task.configure("position", "plugin", frame_parameters("left"), TaskContext{make_model()});
    ASSERT_TRUE(task.set_offset(iso(Eigen::Vector3d(0, 0, 0.2))));
    ASSERT_TRUE(task.set_use_offset(true, message));
    const auto result = task.update(WholeBodyState{}, backend);
    ASSERT_TRUE(result.active);
    EXPECT_NEAR(result.frame_pose.translation().z(), 0.2, 1e-12);
    EXPECT_EQ(result.jacobian.rows(), 3);
  }
}

class PoseOffsetIOTest : public ::testing::Test
{
protected:
  rclcpp::Node::SharedPtr node;
  std::shared_ptr<FramePoseTask> task;
  std::unique_ptr<PoseTaskIO> io;
  rclcpp::Publisher<geometry_msgs::msg::PoseStamped>::SharedPtr publisher;
  rclcpp::executors::SingleThreadedExecutor executor;
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    node = std::make_shared<rclcpp::Node>("pose_offset_io_test", "/pose_offset_test");
    task = std::make_shared<FramePoseTask>();
    task->configure("pose", "plugin", frame_parameters("left"), TaskContext{make_model()});
    io = std::make_unique<PoseTaskIO>();
    io->configure(node.get(), {task}, "world", "base");
    io->set_active(true);
    publisher = node->create_publisher<geometry_msgs::msg::PoseStamped>("task_priority/tasks/pose/offset", 1);
    executor.add_node(node);
    const auto deadline = std::chrono::steady_clock::now() + seconds(2);
    while (publisher->get_subscription_count() < 1 && std::chrono::steady_clock::now() < deadline) {
      executor.spin_some();
    }
    ASSERT_GE(publisher->get_subscription_count(), 1U);
  }
  void TearDown() override
  {
    executor.remove_node(node);
    io.reset();
    task.reset();
    publisher.reset();
    node.reset();
  }
  bool send(const std::string & frame, double z, double qw = 1.0)
  {
    geometry_msgs::msg::PoseStamped message;
    message.header.frame_id = frame;
    message.pose.position.z = z;
    message.pose.orientation.w = qw;
    publisher->publish(message);
    const auto deadline = std::chrono::steady_clock::now() + milliseconds(300);
    while (std::chrono::steady_clock::now() < deadline) {
      executor.spin_some();
      if (task->offset_ready()) {return true;}
      std::this_thread::sleep_for(milliseconds(2));
    }
    return task->offset_ready();
  }
};

TEST_F(PoseOffsetIOTest, ValidOffsetInTheTaskFrameIsStored)
{
  ASSERT_FALSE(task->offset_ready());
  EXPECT_TRUE(send("left", 0.2));
  std::string message;
  EXPECT_TRUE(task->set_use_offset(true, message)) << message;
  auto backend = identity_backend();
  EXPECT_NEAR(task->update(WholeBodyState{}, backend).frame_pose.translation().z(), 0.2, 1e-12);
}

TEST_F(PoseOffsetIOTest, EmptyFrameIdIsAcceptedAsTheTaskFrame)
{
  EXPECT_TRUE(send("", 0.2));
}

TEST_F(PoseOffsetIOTest, OffsetInAnotherFrameIsIgnored)
{
  EXPECT_FALSE(send("right", 0.2));
}

TEST_F(PoseOffsetIOTest, InvalidPosesAreIgnored)
{
  EXPECT_FALSE(send("left", std::numeric_limits<double>::quiet_NaN()));
  EXPECT_FALSE(send("left", 0.2, 0.0));  // zero quaternion
}

TEST_F(PoseOffsetIOTest, UseOffsetStateIsPublishedWhenItChanges)
{
  std::vector<bool> received;
  auto observer = node->create_subscription<std_msgs::msg::Bool>(
    "task_priority/tasks/pose/use_offset", rclcpp::QoS(1).reliable().transient_local(),
    [&received](std_msgs::msg::Bool::SharedPtr msg) {received.push_back(msg->data);});
  const std::vector<std::shared_ptr<TaskBase>> tasks{task};
  const std::vector<TaskComputation> outputs(1);
  auto spin = [this]() {
    const auto deadline = std::chrono::steady_clock::now() + milliseconds(200);
    while (std::chrono::steady_clock::now() < deadline) {
      executor.spin_some();
      std::this_thread::sleep_for(milliseconds(2));
    }
  };
  io->publish_targets(tasks, outputs, node->now(), "world");
  io->publish_targets(tasks, outputs, node->now(), "world");  // unchanged: not republished
  spin();
  ASSERT_EQ(received.size(), 1U);
  EXPECT_FALSE(received.back());
  ASSERT_TRUE(send("left", 0.2));
  std::string message;
  ASSERT_TRUE(task->set_use_offset(true, message));
  io->publish_targets(tasks, outputs, node->now(), "world");
  spin();
  ASSERT_EQ(received.size(), 2U);
  EXPECT_TRUE(received.back());
}

}  // namespace task_priority_kinematic_control
