#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/end_effector_pose_task.hpp"
#include "task_priority_kinematic_control/core/pose_task_io.hpp"
#include "task_priority_kinematic_control/core/manual_task_io.hpp"
#include <gtest/gtest.h>
#include <limits>
#include <chrono>
#include <thread>
#include <cmath>
#include <algorithm>
#include <stdexcept>
#include <future>

namespace task_priority_kinematic_control
{
class PoseBackend : public KinematicsBackend
{
public:
  FrameState frame;
  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState &) override {}
  FrameState get_frame_state(const std::string &) const override {return frame;}
  Eigen::Isometry3d get_relative_transform(const std::string &, const std::string &) const override
  {return Eigen::Isometry3d::Identity();}
  const std::vector<CollisionCapsule> & collision_capsules() const override {return capsules;}
  std::string name() const override {return "pose_test";}
  std::vector<CollisionCapsule> capsules;
};

TEST(FramePoseAxesTest, ComplementaryTasksLeaveOnlyTheirSelectedJacobianRows)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left", "right", {}, {}, {0,1,2,3,4,5});
  PoseBackend backend;
  WholeBodyState state;
  const std::vector<std::vector<int64_t>> masks = {
    {0,0,0,1,1,0}, {1,1,1,0,0,1}};
  for (const auto & mask : masks) {
    FramePoseTask task;
    task.configure("pose", "plugin", {
      {"activation", rclcpp::Parameter("activation", mask)}}, TaskContext{model});
    ASSERT_TRUE(task.update(state, backend).has_updated_target);
    PoseVelocity ff;
    ff << 1,2,3,4,5,6;
    ASSERT_TRUE(task.set_feedforward(ff, task.feedforward_epoch()));
    const auto result = task.update(state, backend);
    const auto rows = static_cast<Eigen::Index>(std::count(mask.begin(), mask.end(), 1));
    ASSERT_EQ(result.jacobian.rows(), rows);
    ASSERT_EQ(result.error.size(), rows);
    ASSERT_EQ(result.desired_velocity.size(), rows);
    Eigen::Index row = 0;
    for (Eigen::Index axis = 0; axis < 6; ++axis) {
      if (mask[axis] == 0) {
        EXPECT_DOUBLE_EQ(result.feedforward(axis), 0);
        EXPECT_TRUE(result.jacobian.col(axis).isZero());
      } else {
        EXPECT_DOUBLE_EQ(result.desired_velocity(row), ff(axis));
        EXPECT_TRUE(result.jacobian.row(row).isApprox(
          Eigen::MatrixXd::Identity(6,6).row(axis)));
        ++row;
      }
    }
  }
}

TEST(FramePoseAxesTest, InactiveFeedforwardDoesNotRecaptureTargetAndInvalidMasksAreRejected)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left", "right", {}, {}, {0,1,2,3,4,5});
  FramePoseTask task;
  auto configure = [&](const std::vector<int64_t> & mask) {
    task.configure("pose", "plugin", {
      {"activation", rclcpp::Parameter("activation", mask)}}, TaskContext{model});
  };
  configure({1,1,1,0,0,1});
  PoseBackend backend;
  WholeBodyState state;
  task.update(state, backend);
  PoseVelocity ff = PoseVelocity::Zero();
  ff(3) = 1;
  ASSERT_TRUE(task.set_feedforward(ff, task.feedforward_epoch()));
  state.base_position.x() = 1;
  const auto result = task.update(state, backend);
  EXPECT_FALSE(result.has_updated_target);
  EXPECT_TRUE(result.feedforward.isZero());
  EXPECT_DOUBLE_EQ(result.error(0), -1);
  EXPECT_THROW(configure({1,0}), std::invalid_argument);
  EXPECT_THROW(configure({1,1,1,2,0,1}), std::invalid_argument);
  configure({0,0,0,0,0,0});
  EXPECT_FALSE(task.update(state, backend).active);
  configure({});
  EXPECT_EQ(task.update(state, backend).jacobian.rows(), 6);
}

TEST(FramePoseTiltTest, LevelsWhenBlockedAndHoldsTiltIndependentlyOfYaw)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left", "right", {}, {}, {0,1,2,3,4,5});
  FramePoseTask task;
  task.configure("stabilize", "plugin", {
    {"activation", rclcpp::Parameter("activation", std::vector<int64_t>{0,0,0,1,1,0})},
    {"roll_pitch_control", rclcpp::Parameter("roll_pitch_control", true)},
    {"feedforward_timeout", rclcpp::Parameter("feedforward_timeout", 0.01)}}, TaskContext{model});
  PoseBackend backend;
  WholeBodyState state;
  state.base_rpy << 0.2, -0.1, 0.8;
  PoseVelocity ff = PoseVelocity::Zero();
  ff(3) = 0.2;
  ASSERT_TRUE(task.set_feedforward(ff, task.feedforward_epoch()));
  auto blocked = task.update(state, backend);
  EXPECT_TRUE(blocked.feedforward.isZero());
  EXPECT_NEAR(blocked.error(0), -0.2, 1e-12);
  EXPECT_NEAR(blocked.error(1), 0.1, 1e-12);
  EXPECT_TRUE(blocked.jacobian.col(5).isZero());
  std::string message;
  ASSERT_TRUE(task.set_allow_roll_pitch(true, message));
  ASSERT_TRUE(task.set_feedforward(ff, task.feedforward_epoch()));
  EXPECT_TRUE(task.update(state, backend).error.isZero());
  state.base_rpy.x() = 0.3;
  ASSERT_TRUE(task.set_feedforward(PoseVelocity::Zero(), task.feedforward_epoch()));
  EXPECT_TRUE(task.update(state, backend).error.isZero());  // Capture on release.
  state.base_rpy.z() = -1.4;
  EXPECT_TRUE(task.update(state, backend).error.isZero());  // Yaw leaves tilt unchanged.
  state.base_rpy.x() = 0.4;
  EXPECT_NEAR(task.update(state, backend).error(0), -0.1, 1e-12);
  ASSERT_TRUE(task.set_feedforward(ff, task.feedforward_epoch()));
  task.update(state, backend);
  state.base_rpy.x() = 0.5;
  std::this_thread::sleep_for(std::chrono::milliseconds(20));
  const auto expired = task.update(state, backend);
  EXPECT_TRUE(expired.feedforward.isZero());
  EXPECT_TRUE(expired.error.isZero());
  state.base_rpy.x() = 0.6;
  EXPECT_NEAR(task.update(state, backend).error(0), -0.1, 1e-12);
  ASSERT_TRUE(task.set_allow_roll_pitch(false, message));
  EXPECT_NEAR(task.update(state, backend).error(0), -0.6, 1e-12);
}

class PoseFeedforwardTest : public ::testing::TestWithParam<int>
{
protected:
  std::shared_ptr<WholeBodyModel> model = std::make_shared<WholeBodyModel>();
  std::unique_ptr<PoseTaskCommon> task;
  PoseBackend backend;
  WholeBodyState state;

  void configure(bool automatic = true, double threshold = 0.001)
  {
    model->configure("world", "base", "tip", "right", {"left_joint"}, {"right_joint"}, {0,1,2,3,4,5});
    task = GetParam() == 2 ? std::unique_ptr<PoseTaskCommon>(new EndEffectorPoseTask) :
      std::unique_ptr<PoseTaskCommon>(new FramePoseTask);
    task->configure("pose", "pose_plugin", {
      {"frame_id", rclcpp::Parameter("frame_id", GetParam() == 0 ? "base" : "tip")},
      {"gain", rclcpp::Parameter("gain", std::vector<double>{2,2,2,2,2,2})},
      {"auto_update_target_pose", rclcpp::Parameter("auto_update_target_pose", automatic)},
      {"feedforward_velocity_threshold", rclcpp::Parameter("feedforward_velocity_threshold", threshold)}},
      TaskContext{model});
    backend.frame.jacobian = Eigen::MatrixXd::Zero(6, model->total_dofs());
  }
  TaskComputation at(double x)
  {
    state.base_position.x() = x;
    backend.frame.pose.translation().x() = x;
    return task->update(state, backend);
  }
  void feedforward(const PoseVelocity & v)
  {ASSERT_TRUE(task->set_feedforward(v, task->feedforward_epoch()));}
};

TEST_P(PoseFeedforwardTest, AddsAllSixAxesToPoseFeedback)
{
  configure(false);
  geometry_msgs::msg::PoseStamped goal;
  goal.pose.position.x = 1;
  goal.pose.position.y = 2;
  goal.pose.position.z = 3;
  goal.pose.orientation.z = std::sin(0.1);
  goal.pose.orientation.w = std::cos(0.1);
  ASSERT_TRUE(task->set_pose_goal(goal));
  PoseVelocity ff;
  ff << 0.1,0.2,0.3,0.4,0.5,0.6;
  feedforward(ff);
  const auto output = at(0);
  PoseVelocity expected;
  expected << 2.1,4.2,6.3,0.4,0.5,1.0;
  EXPECT_TRUE(output.desired_velocity.isApprox(expected, 1e-10));
  EXPECT_FALSE(output.has_updated_target);
  EXPECT_TRUE(output.feedforward.isApprox(ff));
}

TEST_P(PoseFeedforwardTest, CapturesAndFollowsOnlyAboveThreshold)
{
  configure(true, 0.1);
  EXPECT_TRUE(at(0).has_updated_target);
  PoseVelocity ff = PoseVelocity::Zero();
  ff.x() = 0.2;
  feedforward(ff);
  auto moving = at(1);
  EXPECT_TRUE(moving.has_updated_target);
  EXPECT_DOUBLE_EQ(moving.updated_target.position.x, 1);
  EXPECT_TRUE(moving.error.isZero());
  ff.x() = 0.1;
  feedforward(ff);
  const auto equal = at(2);
  EXPECT_FALSE(equal.has_updated_target);
  EXPECT_DOUBLE_EQ(task->current_target()[0], 1);
  EXPECT_NEAR(equal.desired_velocity(0), -1.9, 1e-12);
  ff.x() = 0.01;
  feedforward(ff);
  EXPECT_FALSE(at(3).has_updated_target);
  feedforward(PoseVelocity::Zero());
  EXPECT_FALSE(at(4).has_updated_target);
  EXPECT_DOUBLE_EQ(task->current_target()[0], 1);
}

TEST_P(PoseFeedforwardTest, NormIncludesRotationAndFeedforwardDoesNotExpire)
{
  configure(true, 0.1);
  at(0);
  PoseVelocity ff = PoseVelocity::Zero();
  ff.tail<3>().setConstant(0.07);  // Each component is below the threshold; the norm is above it.
  feedforward(ff);
  for (int i = 1; i < 100; ++i) {
    const auto output = at(i);
    EXPECT_TRUE(output.has_updated_target);
    EXPECT_TRUE(output.desired_velocity.isApprox(ff));
  }
}

TEST_P(PoseFeedforwardTest, DisableAndResetRejectPreviousEpochs)
{
  configure();
  const auto epoch = task->feedforward_epoch();
  feedforward(PoseVelocity::Ones());
  ASSERT_TRUE(task->set_enabled(false));
  EXPECT_FALSE(task->set_feedforward(PoseVelocity::Ones(), task->feedforward_epoch()));
  ASSERT_TRUE(task->set_enabled(true));
  EXPECT_FALSE(task->set_feedforward(PoseVelocity::Ones(), epoch));
  EXPECT_TRUE(at(5).feedforward.isZero());
  EXPECT_DOUBLE_EQ(task->current_target()[0], 5);
  feedforward(PoseVelocity::Ones());
  const auto before_reset = task->feedforward_epoch();
  task->reset();
  EXPECT_FALSE(task->set_feedforward(PoseVelocity::Ones(), before_reset));
  EXPECT_TRUE(at(6).feedforward.isZero());
}

TEST_P(PoseFeedforwardTest, InvalidInputClearsVelocityAndInvalidThresholdIsRejected)
{
  configure();
  feedforward(PoseVelocity::Ones());
  PoseVelocity bad = PoseVelocity::Zero();
  bad(3) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(task->set_feedforward(bad, task->feedforward_epoch()));
  EXPECT_TRUE(at(0).feedforward.isZero());
  EXPECT_THROW(configure(true, -0.1), std::invalid_argument);
  EXPECT_THROW(configure(true, std::numeric_limits<double>::infinity()), std::invalid_argument);
}

TEST_P(PoseFeedforwardTest, OptionsCanBeChangedWithoutRestart)
{
  configure(false);
  std::string message;
  ASSERT_TRUE(task->set_feedforward_option("auto_update_target_pose",
    rclcpp::Parameter("auto_update_target_pose", true), message));
  EXPECT_TRUE(at(4).has_updated_target);
  ASSERT_TRUE(task->set_feedforward_option("feedforward_velocity_threshold",
    rclcpp::Parameter("feedforward_velocity_threshold", 2.0), message));
  feedforward(PoseVelocity::UnitX());
  EXPECT_FALSE(at(5).has_updated_target);
  EXPECT_FALSE(task->set_feedforward_option("feedforward_velocity_threshold",
    rclcpp::Parameter("feedforward_velocity_threshold", -1.0), message));
  EXPECT_FALSE(task->set_feedforward_option("auto_update_target_pose",
    rclcpp::Parameter("auto_update_target_pose", 1.0), message));
}

TEST_P(PoseFeedforwardTest, InvalidPoseDoesNotInitializeTarget)
{
  configure();
  EXPECT_FALSE(at(std::numeric_limits<double>::quiet_NaN()).active);
  EXPECT_TRUE(task->current_target().empty());
  EXPECT_TRUE(at(7).has_updated_target);
  EXPECT_DOUBLE_EQ(task->current_target()[0], 7);
}

INSTANTIATE_TEST_SUITE_P(BaseAndEndEffectors, PoseFeedforwardTest, ::testing::Values(0, 1, 2));

TEST(PoseVelocityRotation, RotatesBothVectorsWithoutChangingTheirPoint)
{
  geometry_msgs::msg::Twist twist;
  twist.linear.x = 1;
  twist.angular.y = 2;
  const Eigen::Quaterniond rotation(Eigen::AngleAxisd(std::acos(-1.0)/2, Eigen::Vector3d::UnitZ()));
  const auto rotated = rotate_pose_velocity(twist, rotation);
  PoseVelocity expected;
  expected << 0,1,0,-2,0,0;
  EXPECT_TRUE(rotated.isApprox(expected, 1e-12));
}

class PoseTaskIOTest : public ::testing::Test
{
protected:
  rclcpp::Node::SharedPtr node;
  std::shared_ptr<WholeBodyModel> model;
  std::shared_ptr<FramePoseTask> task;
  std::unique_ptr<PoseTaskIO> io;
  rclcpp::Publisher<geometry_msgs::msg::TwistStamped>::SharedPtr publisher;
  rclcpp::executors::SingleThreadedExecutor executor;
  PoseBackend backend;
  WholeBodyState state;
  int received{0};
  rclcpp::Subscription<geometry_msgs::msg::TwistStamped>::SharedPtr observer;
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}

  void SetUp() override
  {
    node = std::make_shared<rclcpp::Node>("pose_feedforward_io_test", "/pose_feedforward_test");
    model = std::make_shared<WholeBodyModel>();
    model->configure("world", "base", "left", "right", {}, {}, {0,1,2,3,4,5});
    task = std::make_shared<FramePoseTask>();
    task->configure("pose", "pose_plugin", {{"frame_id", rclcpp::Parameter("frame_id", "base")}}, TaskContext{model});
    io = std::make_unique<PoseTaskIO>();
    io->configure(node.get(), {task}, "world", "base");
    io->set_active(true);
    publisher = node->create_publisher<geometry_msgs::msg::TwistStamped>("task_priority/tasks/pose/feedforward", 1);
    observer = node->create_subscription<geometry_msgs::msg::TwistStamped>(
      "task_priority/tasks/pose/feedforward", 1,
      [this](geometry_msgs::msg::TwistStamped::SharedPtr) {++received;});
    executor.add_node(node);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (publisher->get_subscription_count() < 2 && std::chrono::steady_clock::now() < deadline) {executor.spin_some();}
    ASSERT_GE(publisher->get_subscription_count(), 2U);
  }
  void TearDown() override
  {
    executor.remove_node(node);
    io.reset(); task.reset(); observer.reset(); publisher.reset(); node.reset();
  }
  void transform(const std::string & frame, double angle, int seconds = 0, bool is_static = true)
  {
    geometry_msgs::msg::TransformStamped t;
    t.header.frame_id = "world";
    t.child_frame_id = frame;
    t.header.stamp.sec = seconds;
    t.transform.translation.x = 100;  // Reference origin must not affect velocity.
    t.transform.rotation.z = std::sin(angle/2);
    t.transform.rotation.w = std::cos(angle/2);
    ASSERT_TRUE(io->buffer_->setTransform(t, "test", is_static));
  }
  size_t target_subscription_count() const
  {
    return io->publishers_.at("pose")->get_subscription_count();
  }
  PoseVelocity send(const std::string & frame, int seconds = 0, double speed = 1)
  {
    geometry_msgs::msg::TwistStamped message;
    message.header.frame_id = frame;
    message.header.stamp.sec = seconds;
    message.twist.linear.x = speed;
    message.twist.angular.x = speed;
    const int before = received;
    publisher->publish(message);
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(1);
    while (received == before && std::chrono::steady_clock::now() < deadline) {executor.spin_some();}
    // Both local subscriptions receive the same sample; wait for callbacks
    // that may become ready just after the observer's callback.
    const auto drain_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(20);
    while (std::chrono::steady_clock::now() < drain_deadline) {
      executor.spin_some();
      std::this_thread::sleep_for(std::chrono::milliseconds(1));
    }
    EXPECT_GT(received, before);
    return task->update(state, backend).feedforward;
  }
};

TEST_F(PoseTaskIOTest, EmptyFrameUsesBaseAndUnknownFrameClearsPreviousVelocity)
{
  transform("base", std::acos(-1.0)/2);
  const auto velocity = send("");
  PoseVelocity expected;
  expected << 0,1,0,0,1,0;
  EXPECT_TRUE(velocity.isApprox(expected, 1e-12));
  EXPECT_TRUE(send("unknown").isZero());
  EXPECT_TRUE(send("world", 0, std::numeric_limits<double>::quiet_NaN()).isZero());
}

TEST_F(PoseTaskIOTest, ManualInterfacesExposeTiltServicesAndAtomicNominalCommands)
{
  model->configure("world", "base", "left", "right", {"a", "b"}, {}, {0,1,2,3,4,5});
  auto tilt = std::make_shared<FramePoseTask>();
  tilt->configure("stabilize", "plugin", {
    {"roll_pitch_control", rclcpp::Parameter("roll_pitch_control", true)},
    {"activation", rclcpp::Parameter("activation", std::vector<int64_t>{0,0,0,1,1,0})}}, TaskContext{model});
  auto nominal = std::make_shared<JointNominalTask>();
  nominal->configure("nominal", "plugin", {}, TaskContext{model});
  std::mutex mutex;
  ManualTaskIO manual;
  manual.configure(node.get(), {tilt, nominal}, mutex);
  manual.set_active(true);
  auto enable = node->create_client<std_srvs::srv::Trigger>("task_priority/tasks/stabilize/enable_roll_pitch");
  auto disable = node->create_client<std_srvs::srv::Trigger>("task_priority/tasks/stabilize/disable_roll_pitch");
  ASSERT_TRUE(enable->wait_for_service(std::chrono::seconds(2)));
  ASSERT_TRUE(disable->wait_for_service(std::chrono::seconds(2)));
  auto call = [&](const rclcpp::Client<std_srvs::srv::Trigger>::SharedPtr & client) {
    auto future = client->async_send_request(std::make_shared<std_srvs::srv::Trigger::Request>());
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
    while (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready &&
      std::chrono::steady_clock::now() < deadline) {executor.spin_some();}
    if (future.wait_for(std::chrono::seconds(0)) != std::future_status::ready) {return false;}
    return future.get()->success;
  };
  state.base_rpy.x() = 0.2;
  EXPECT_NEAR(tilt->update(state, backend).error(0), -0.2, 1e-12);
  ASSERT_TRUE(call(enable));
  EXPECT_TRUE(tilt->update(state, backend).error.isZero());
  ASSERT_TRUE(call(disable));
  EXPECT_NEAR(tilt->update(state, backend).error(0), -0.2, 1e-12);
  auto command_pub = node->create_publisher<sensor_msgs::msg::JointState>("task_priority/tasks/nominal/joint_command", 1);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (command_pub->get_subscription_count() == 0 && std::chrono::steady_clock::now() < deadline) {executor.spin_some();}
  ASSERT_GT(command_pub->get_subscription_count(), 0U);
  sensor_msgs::msg::JointState command;
  command.name = {"a", "b"}; command.position = {1,2}; command.velocity = {0.1,0.2};
  command_pub->publish(command);
  const auto until = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while (nominal->current_target() != command.position && std::chrono::steady_clock::now() < until) {executor.spin_some();}
  EXPECT_EQ(nominal->current_target(), command.position);
  state.joint_positions = Eigen::Vector2d(1,2);
  EXPECT_NEAR(nominal->update(state, backend).desired_velocity(0), 0.1, 1e-12);
  manual.set_active(false);
  EXPECT_TRUE(nominal->update(state, backend).feedforward.isZero());
}

TEST_F(PoseTaskIOTest, HonorsTimestampAndZeroUsesLatestTransform)
{
  transform("external", 0, 10, false);
  transform("external", std::acos(-1.0)/2, 20, false);
  EXPECT_NEAR(send("external", 10).x(), 1, 1e-12);
  EXPECT_NEAR(send("external", 20).y(), 1, 1e-12);
  EXPECT_NEAR(send("external").y(), 1, 1e-12);
  EXPECT_TRUE(send("external", 30).isZero());
}

TEST_F(PoseTaskIOTest, OnlyOwnTargetsAreIgnoredIncludingWithLocalExternalPublishers)
{
  int targets = 0;
  auto subscription = node->create_subscription<geometry_msgs::msg::PoseStamped>(
    "task_priority/tasks/pose/target", 10,
    [this, &targets](geometry_msgs::msg::PoseStamped::SharedPtr goal, const rclcpp::MessageInfo & info) {
      if (!io->is_own_target("pose", info)) {task->set_pose_goal(*goal); ++targets;}
    });
  auto external = node->create_publisher<geometry_msgs::msg::PoseStamped>("task_priority/tasks/pose/target", 10);
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(2);
  while ((external->get_subscription_count() == 0 ||
    target_subscription_count() == 0) &&
    std::chrono::steady_clock::now() < deadline) {executor.spin_some();}
  ASSERT_GT(external->get_subscription_count(), 0U);
  ASSERT_GT(target_subscription_count(), 0U);
  auto output = task->update(state, backend);
  ASSERT_TRUE(output.has_updated_target);
  io->publish_targets({task}, {output}, node->now(), "world");
  auto drain = [&]() {
    const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(50);
    while (std::chrono::steady_clock::now() < until) {executor.spin_some();}
  };
  drain();
  EXPECT_EQ(targets, 0);
  geometry_msgs::msg::PoseStamped goal;
  goal.pose.position.x = 9;
  goal.pose.orientation.w = 1;
  external->publish(goal);
  drain();
  EXPECT_EQ(targets, 1);
  EXPECT_DOUBLE_EQ(task->current_target()[0], 9);
}

TEST_F(PoseTaskIOTest, LifecycleClearsVelocityAndIgnoresInactiveMessages)
{
  EXPECT_DOUBLE_EQ(send("world").x(), 1);
  io->set_active(false);
  EXPECT_TRUE(send("world").isZero());
  io->set_active(true);
  EXPECT_TRUE(task->update(state, backend).feedforward.isZero());
}
}  // namespace task_priority_kinematic_control
