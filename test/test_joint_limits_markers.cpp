#include "task_priority_kinematic_control/core/capsule_observer.hpp"
#include "task_priority_kinematic_control/kinematics/kdl_kinematics_backend.hpp"
#include <gtest/gtest.h>
#include <chrono>
#include <cmath>
#include <functional>
#include <limits>
#include <map>
#include <thread>

namespace task_priority_kinematic_control
{
namespace
{
using Marker = visualization_msgs::msg::Marker;
using Params = std::map<std::string, rclcpp::Parameter>;

std::shared_ptr<WholeBodyModel> marker_model()
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "link_c", "link_c",
    {"joint_a", "joint_b", "joint_c"}, {}, {0, 1, 2, 3, 4, 5});
  return model;
}

Params marker_parameters(bool publish = true, double rate = 1.0)
{
  return {
    {"enabled", rclcpp::Parameter("enabled", true)},
    {"alpha", rclcpp::Parameter("alpha", .1)},
    {"delta", rclcpp::Parameter("delta", .15)},
    {"eps", rclcpp::Parameter("eps", .0001)},
    {"gain_scalar", rclcpp::Parameter("gain_scalar", .5)},
    {"publish_limits_marker", rclcpp::Parameter("publish_limits_marker", publish)},
    {"limits_marker_publish_rate", rclcpp::Parameter("limits_marker_publish_rate", rate)},
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{0, 0, 0})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{1, 1, 1})}
  };
}

class MarkerBackend : public KinematicsBackend
{
public:
  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState &) override {}
  FrameState get_frame_state(const std::string & frame) const override {return frames.at(frame);}
  bool read_frame_state(const std::string & frame, FrameState & out) const override
  {
    ++pose_reads;
    const auto it = frames.find(frame);
    if (it == frames.end()) {return false;}
    out.pose = it->second.pose; out.jacobian = it->second.jacobian;
    return true;
  }
  bool joint_child_frame(const std::string & joint, std::string & frame) const override
  {
    ++topology_reads;
    const auto it = children.find(joint);
    if (it == children.end()) {frame.clear(); return false;}
    frame = it->second; return true;
  }
  Eigen::Isometry3d get_relative_transform(const std::string & a, const std::string & b) const override
  {return frames.at(a).pose.inverse() * frames.at(b).pose;}
  const std::vector<CollisionCapsule> & collision_capsules() const override {return capsules;}
  std::string name() const override {return "marker_mock";}
  std::map<std::string, FrameState> frames;
  std::map<std::string, std::string> children;
  std::vector<CollisionCapsule> capsules;
  mutable size_t pose_reads = 0;
  mutable size_t topology_reads = 0;
};

std::shared_ptr<MarkerBackend> marker_backend()
{
  auto backend = std::make_shared<MarkerBackend>();
  Eigen::Isometry3d base = Eigen::Isometry3d::Identity();
  base.translation() = Eigen::Vector3d(2, 3, 4);
  base.linear() = Eigen::AngleAxisd(.7, Eigen::Vector3d(1, 2, 3).normalized()).toRotationMatrix();
  FrameState frame; frame.pose = base; frame.jacobian = Eigen::MatrixXd::Zero(6, 9);
  backend->frames["base"] = frame;
  for (const auto & entry : std::vector<std::pair<std::string, std::string>>{
      {"joint_a", "link_a"}, {"joint_b", "link_b"}, {"joint_c", "link_c"}}) {
    const double index = static_cast<double>(backend->children.size());
    backend->children[entry.first] = entry.second;
    frame.pose = base;
    frame.pose.translation() += base.rotation() * Eigen::Vector3d(.1 + index, .2, .3);
    frame.jacobian.setZero();
    frame.jacobian.block<3, 1>(3, 6 + static_cast<Eigen::Index>(index)) =
      base.rotation() * Eigen::Vector3d::UnitZ();
    backend->frames[entry.second] = frame;
  }
  return backend;
}

std::shared_ptr<JointLimitsTask> marker_task(
  const Params & parameters, const KinematicsBackendPtr & backend)
{
  auto task = std::make_shared<JointLimitsTask>();
  task->configure("joint_limits", "task_priority_kinematic_control/JointLimitsTask", parameters,
    TaskContext{marker_model(), backend});
  return task;
}

TEST(JointLimitsMarkers, PositionsIdentityHysteresisAndRelease)
{
  auto backend = marker_backend(); auto task = marker_task(marker_parameters(), backend);
  EXPECT_EQ(backend->topology_reads, 3u);
  WholeBodyState state; state.joint_positions = Eigen::Vector3d(.05, .95, .5);
  TaskComputation output; task->prepare_computation(output); task->update_into(state, *backend, output);
  EXPECT_EQ(backend->pose_reads, 3u); // Base plus the two active joints only.
  task->observe_command({}, 1230000000);
  const auto * snapshot = task->consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  ASSERT_EQ(snapshot->joints.size(), 3u);
  EXPECT_TRUE(snapshot->joints[0].active); EXPECT_TRUE(snapshot->joints[1].active);
  EXPECT_FALSE(snapshot->joints[2].active); EXPECT_FALSE(snapshot->joints[2].valid);
  EXPECT_LT((snapshot->joints[0].position - Eigen::Vector3d(.1, .2, .3)).norm(), 1e-12);
  EXPECT_LT((snapshot->joints[1].position - Eigen::Vector3d(1.1, .2, .3)).norm(), 1e-12);
  auto markers = joint_limits_markers(task->joint_names(), *snapshot, task->base_frame(), task->id());
  ASSERT_EQ(markers.markers.size(), 3u);
  for (size_t i = 0; i < markers.markers.size(); ++i) {
    const auto & m = markers.markers[i];
    EXPECT_EQ(m.id, static_cast<int>(i)); EXPECT_EQ(m.type, Marker::CYLINDER);
    EXPECT_EQ(m.ns, "joint_limits/limits/" + task->joint_names()[i]);
    EXPECT_EQ(m.header.frame_id, "base"); EXPECT_EQ(m.header.stamp.sec, 1);
    EXPECT_EQ(m.header.stamp.nanosec, 230000000u); EXPECT_TRUE(m.text.empty());
    EXPECT_DOUBLE_EQ(m.scale.x, .06); EXPECT_DOUBLE_EQ(m.scale.y, .06); EXPECT_DOUBLE_EQ(m.scale.z, .07);
    EXPECT_FLOAT_EQ(m.color.r, 1); EXPECT_FLOAT_EQ(m.color.g, 0); EXPECT_FLOAT_EQ(m.color.b, 1);
    EXPECT_FLOAT_EQ(m.color.a, 1); EXPECT_DOUBLE_EQ(m.pose.orientation.w, 1);
    EXPECT_EQ(m.action, i < 2 ? Marker::ADD : Marker::DELETE);
  }
  state.joint_positions = Eigen::Vector3d(.12, .88, .5);
  task->update_into(state, *backend, output); task->observe_command({}, 2);
  snapshot = task->consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_TRUE(snapshot->joints[0].active); EXPECT_TRUE(snapshot->joints[1].active);
  state.joint_positions(0) = .14995;
  task->update_into(state, *backend, output); task->observe_command({}, 3);
  snapshot = task->consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_FALSE(snapshot->joints[0].active); EXPECT_TRUE(snapshot->joints[1].active);
  markers = joint_limits_markers(task->joint_names(), *snapshot, "base", task->id());
  EXPECT_EQ(markers.markers[0].action, Marker::DELETE); EXPECT_EQ(markers.markers[1].action, Marker::ADD);
  EXPECT_EQ(backend->topology_reads, 3u); // Configuration lookup is never repeated by update.
}

TEST(JointLimitsMarkers, CylinderFollowsJointAxisWithRotatedBaseAndChildFrame)
{
  auto backend = marker_backend(); auto task = marker_task(marker_parameters(), backend);
  const Eigen::Matrix3d base_rotation = backend->frames.at("base").pose.rotation();
  auto & child = backend->frames.at("link_a");
  child.pose.linear() = base_rotation *
    Eigen::AngleAxisd(.8, Eigen::Vector3d::UnitY()).toRotationMatrix();
  WholeBodyState state; state.joint_positions = Eigen::Vector3d(.05, .5, .5);
  TaskComputation output; task->prepare_computation(output);
  const std::vector<Eigen::Vector3d> axes{
    Eigen::Vector3d::UnitX(), Eigen::Vector3d::UnitY(),
    -Eigen::Vector3d::UnitZ(), Eigen::Vector3d(1, 2, 3).normalized()};
  for (const auto & axis : axes) {
    for (bool prismatic : {false, true}) {
      child.jacobian.setZero();
      child.jacobian.block<3, 1>(prismatic ? 0 : 3, 6) = 2 * base_rotation * axis;
      task->update_into(state, *backend, output); task->observe_command({}, 1);
      const auto * snapshot = task->consume_snapshot(); ASSERT_NE(snapshot, nullptr);
      ASSERT_TRUE(snapshot->joints[0].valid);
      EXPECT_LT((snapshot->joints[0].axis - axis).norm(), 1e-12);
      const auto markers = joint_limits_markers(task->joint_names(), *snapshot, "base", task->id());
      const auto & marker = markers.markers[0]; ASSERT_EQ(marker.action, Marker::ADD);
      const auto & q = marker.pose.orientation;
      const Eigen::Quaterniond orientation(q.w, q.x, q.y, q.z);
      EXPECT_NEAR(orientation.norm(), 1, 1e-12);
      EXPECT_LT((orientation * Eigen::Vector3d::UnitZ() - axis).norm(), 1e-12);
      EXPECT_DOUBLE_EQ(marker.scale.x, .06); EXPECT_DOUBLE_EQ(marker.scale.y, .06);
      EXPECT_DOUBLE_EQ(marker.scale.z, .07);
    }
  }
  EXPECT_EQ(backend->topology_reads, 3u);
  EXPECT_EQ(backend->pose_reads, 2u * axes.size() * 2u);
}

TEST(JointLimitsMarkers, UnavailableJointAxisOnlyDeletesMarker)
{
  auto backend = marker_backend(); auto task = marker_task(marker_parameters(), backend);
  WholeBodyState state; state.joint_positions = Eigen::Vector3d(.05, .5, .5);
  TaskComputation output; task->prepare_computation(output);
  auto & jacobian = backend->frames.at("link_a").jacobian;
  const auto capture = [&] {
      task->update_into(state, *backend, output); task->observe_command({}, 1);
      const auto * snapshot = task->consume_snapshot(); EXPECT_NE(snapshot, nullptr);
      EXPECT_TRUE(output.active); EXPECT_DOUBLE_EQ(output.jacobian(0, 6), 1);
      EXPECT_FALSE(output.stop_arm_motion);
      return snapshot;
    };
  ASSERT_TRUE(capture()->joints[0].valid);
  jacobian.setZero();
  EXPECT_FALSE(capture()->joints[0].valid);
  jacobian(3, 6) = std::numeric_limits<double>::quiet_NaN();
  auto snapshot = capture(); EXPECT_FALSE(snapshot->joints[0].valid);
  EXPECT_EQ(joint_limits_markers(task->joint_names(), *snapshot, "base", task->id()).markers[0].action,
    Marker::DELETE);
  jacobian.setZero(6, 6);
  EXPECT_FALSE(capture()->joints[0].valid);
  jacobian.setZero(6, 9); jacobian.block<3, 1>(3, 6) =
    backend->frames.at("base").pose.rotation() * Eigen::Vector3d::UnitZ();
  EXPECT_TRUE(capture()->joints[0].valid);
}

TEST(JointLimitsMarkers, DisabledPublicationHasNoPoseReadsAndZeroGainStillShowsLatchedLimits)
{
  auto backend = marker_backend(); auto parameters = marker_parameters(false);
  auto task = marker_task(parameters, backend);
  WholeBodyState state; state.joint_positions = Eigen::Vector3d(.05, .5, .5);
  TaskComputation output; task->prepare_computation(output);
  for (int i = 0; i < 1000; ++i) {task->update_into(state, *backend, output); task->observe_command({}, i);}
  EXPECT_EQ(backend->pose_reads, 0u); EXPECT_EQ(task->consume_snapshot(), nullptr);
  task->set_publish_limits_marker(true);
  std::string message; ASSERT_TRUE(task->set_gain_scalar(0, message));
  const double * storage = output.jacobian.data();
  task->update_into(state, *backend, output); task->observe_command({}, 1001);
  EXPECT_FALSE(output.active); EXPECT_DOUBLE_EQ(output.jacobian.norm(), 0);
  EXPECT_EQ(output.jacobian.data(), storage); EXPECT_EQ(backend->pose_reads, 2u);
  const auto * snapshot = task->consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_TRUE(snapshot->joints[0].active); EXPECT_TRUE(snapshot->joints[0].valid);
  task->set_publish_limits_marker(false); task->update_into(state, *backend, output);
  EXPECT_EQ(backend->pose_reads, 2u);
}

TEST(JointLimitsMarkers, PreparedSnapshotStorageIsReusedAndUnsupportedBackendDoesNotAffectControl)
{
  auto backend = marker_backend(); auto task = marker_task(marker_parameters(), backend);
  TaskComputation output; task->prepare_computation(output);
  const double * storage = output.jacobian.data();
  const char * status = output.status_message.data();
  const JointLimitPose * slots[3] = {nullptr, nullptr, nullptr};
  WholeBodyState state; state.joint_positions = Eigen::Vector3d(.05, .5, .5);
  for (int iteration = 0; iteration < 1000; ++iteration) {
    task->update_into(state, *backend, output); task->observe_command({}, iteration);
    const auto * snapshot = task->consume_snapshot(); ASSERT_NE(snapshot, nullptr);
    ASSERT_EQ(snapshot->joints.size(), 3u); EXPECT_TRUE(snapshot->joints[0].valid);
    bool found = false;
    for (auto & slot : slots) {
      if (slot == snapshot->joints.data()) {found = true; break;}
      if (!slot) {slot = snapshot->joints.data(); found = true; break;}
    }
    EXPECT_TRUE(found); EXPECT_EQ(output.jacobian.data(), storage); EXPECT_EQ(output.status_message.data(), status);
  }
  EXPECT_EQ(backend->topology_reads, 3u);
  std::string frame = "stale";
  EXPECT_FALSE(backend->KinematicsBackend::joint_child_frame("joint_a", frame)); EXPECT_TRUE(frame.empty());
  backend->children.clear(); auto unsupported = marker_task(marker_parameters(), backend);
  const size_t reads = backend->pose_reads;
  unsupported->update_into(state, *backend, output); unsupported->observe_command({}, 1);
  EXPECT_TRUE(output.active); EXPECT_FALSE(output.stop_arm_motion); EXPECT_EQ(backend->pose_reads, reads);
  const auto * snapshot = unsupported->consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_TRUE(snapshot->joints[0].active); EXPECT_FALSE(snapshot->joints[0].valid);
}

TEST(JointLimitsMarkers, MissingInvalidFramesAndInvalidStatesOnlyDeleteMarkers)
{
  auto backend = marker_backend(); backend->children.erase("joint_c");
  auto task = marker_task(marker_parameters(), backend);
  WholeBodyState state; state.joint_positions = Eigen::Vector3d(.05, .95, .05);
  TaskComputation output; task->prepare_computation(output);
  const auto capture = [&] {
      task->update_into(state, *backend, output); task->observe_command({}, 1);
      return task->consume_snapshot();
    };
  auto snapshot = capture(); ASSERT_NE(snapshot, nullptr);
  EXPECT_TRUE(output.active); EXPECT_FALSE(output.stop_arm_motion);
  EXPECT_TRUE(snapshot->joints[0].valid); EXPECT_TRUE(snapshot->joints[1].valid);
  EXPECT_TRUE(snapshot->joints[2].active); EXPECT_FALSE(snapshot->joints[2].valid);
  backend->frames["link_a"].pose.translation().x() = std::numeric_limits<double>::quiet_NaN();
  snapshot = capture(); ASSERT_NE(snapshot, nullptr);
  EXPECT_FALSE(snapshot->joints[0].valid); EXPECT_TRUE(snapshot->joints[1].valid);
  EXPECT_TRUE(output.active); EXPECT_FALSE(output.stop_arm_motion);
  backend->frames.erase("link_b"); snapshot = capture(); ASSERT_NE(snapshot, nullptr);
  EXPECT_FALSE(snapshot->joints[1].valid); EXPECT_DOUBLE_EQ(output.jacobian(1, 7), 1);
  backend->frames.erase("base"); snapshot = capture(); ASSERT_NE(snapshot, nullptr);
  for (const auto & joint : snapshot->joints) {EXPECT_FALSE(joint.valid);}
  for (const Eigen::Index count : {0, 2, 4}) {
    state.joint_positions.setConstant(count, .05); snapshot = capture(); ASSERT_NE(snapshot, nullptr);
    EXPECT_FALSE(snapshot->enabled); EXPECT_FALSE(output.active); EXPECT_FALSE(output.stop_arm_motion);
    for (const auto & joint : snapshot->joints) {EXPECT_FALSE(joint.valid); EXPECT_FALSE(joint.active);}
  }
  task->set_enabled(false); state.joint_positions.setConstant(3, .05);
  snapshot = capture(); ASSERT_NE(snapshot, nullptr); EXPECT_FALSE(snapshot->enabled);
  const auto clear = delete_joint_limits_markers(task->joint_names(), "base", task->id());
  ASSERT_EQ(clear.markers.size(), 3u);
  for (const auto & m : clear.markers) {
    EXPECT_EQ(m.action, Marker::DELETE); EXPECT_NE(m.ns.find("joint_limits/limits/"), std::string::npos);
  }
}

TEST(JointLimitsMarkers, KDLChildFramesAndMovingBaseIncludePrismaticJoint)
{
  const std::string urdf = R"(<robot name="markers"><link name="base"/>
    <link name="link_a"/><link name="link_b"/><link name="link_c"/>
    <joint name="joint_a" type="revolute"><parent link="base"/><child link="link_a"/>
      <origin xyz="0.1 0 0"/><axis xyz="0 0 1"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
    <joint name="joint_b" type="prismatic"><parent link="link_a"/><child link="link_b"/>
      <origin xyz="0.2 0 0"/><axis xyz="1 0 0"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
    <joint name="joint_c" type="revolute"><parent link="link_b"/><child link="link_c"/>
      <origin xyz="0.3 0 0"/><axis xyz="0 0 1"/><limit lower="0" upper="1" effort="1" velocity="1"/></joint>
    </robot>)";
  auto model = marker_model(); auto backend = std::make_shared<KDLKinematicsBackend>();
  backend->configure(*model, urdf, rclcpp::get_logger("joint_limit_markers_kdl"));
  std::string frame = "stale";
  ASSERT_TRUE(backend->joint_child_frame("joint_a", frame)); EXPECT_EQ(frame, "link_a");
  ASSERT_TRUE(backend->joint_child_frame("joint_b", frame)); EXPECT_EQ(frame, "link_b");
  ASSERT_TRUE(backend->clone()->joint_child_frame("joint_c", frame)); EXPECT_EQ(frame, "link_c");
  EXPECT_FALSE(backend->joint_child_frame("missing", frame)); EXPECT_TRUE(frame.empty());
  auto task = marker_task(marker_parameters(), backend);
  WholeBodyState state; state.joint_positions = Eigen::Vector3d(.05, .95, .5);
  state.joint_velocities = Eigen::Vector3d::Zero(); backend->update(state);
  TaskComputation output; task->prepare_computation(output);
  task->update_into(state, *backend, output); task->observe_command({}, 1);
  const auto * snapshot = task->consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  const auto first = snapshot->joints;
  EXPECT_LT((first[0].axis - Eigen::Vector3d::UnitZ()).norm(), 1e-12);
  EXPECT_LT((first[1].axis - Eigen::Vector3d(std::cos(.05), std::sin(.05), 0)).norm(), 1e-12);
  EXPECT_LT((first[0].position - Eigen::Vector3d(.1, 0, 0)).norm(), 1e-12);
  EXPECT_LT((first[1].position - Eigen::Vector3d(.1 + 1.15 * std::cos(.05), 1.15 * std::sin(.05), 0)).norm(), 1e-12);
  state.base_position = Eigen::Vector3d(3, 2, 1); state.base_rpy = Eigen::Vector3d(.3, -.4, .7);
  backend->update(state); task->update_into(state, *backend, output); task->observe_command({}, 2);
  snapshot = task->consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  for (size_t i = 0; i < 2; ++i) {
    EXPECT_TRUE(snapshot->joints[i].valid); EXPECT_LT((snapshot->joints[i].position - first[i].position).norm(), 1e-12);
    EXPECT_LT((snapshot->joints[i].axis - first[i].axis).norm(), 1e-12);
  }
}

TEST(JointLimitsMarkers, InvalidRatesAndAtomicParameterValidation)
{
  auto backend = marker_backend();
  for (double rate : {0.0, -1.0, std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(marker_task(marker_parameters(false, rate), backend), std::invalid_argument);
  }
  auto task = marker_task(marker_parameters(false), backend);
  std::vector<std::shared_ptr<TaskBase>> tasks{task};
  const std::string prefix = "tasks.joint_limits.";
  EXPECT_FALSE(validate_joint_limits_marker_parameter_updates(
    {rclcpp::Parameter(prefix + "publish_limits_marker", "true")}, tasks).successful);
  EXPECT_FALSE(validate_joint_limits_marker_parameter_updates(
    {rclcpp::Parameter(prefix + "publish_limits_marker", true),
      rclcpp::Parameter(prefix + "limits_marker_publish_rate", 2.0)}, tasks).successful);
  EXPECT_FALSE(task->publish_limits_marker());
  EXPECT_TRUE(validate_joint_limits_marker_parameter_updates(
    {rclcpp::Parameter(prefix + "publish_limits_marker", true)}, tasks, false).successful);
  EXPECT_FALSE(task->publish_limits_marker());
  EXPECT_TRUE(validate_joint_limits_marker_parameter_updates(
    {rclcpp::Parameter(prefix + "publish_limits_marker", true)}, tasks).successful);
  EXPECT_TRUE(task->publish_limits_marker());
}

class JointLimitsObserverTest : public ::testing::Test
{
  void SetUp() override {if (!rclcpp::ok()) {rclcpp::init(0, nullptr);}}
};

TEST_F(JointLimitsObserverTest, DeclaresDefaultsAndOverrides)
{
  const std::string prefix = "tasks.joint_limits.";
  auto node = std::make_shared<rclcpp::Node>("limits_marker_defaults");
  declare_joint_limits_marker_parameters(node->get_node_parameters_interface(), prefix);
  EXPECT_FALSE(node->get_parameter(prefix + "publish_limits_marker").as_bool());
  EXPECT_DOUBLE_EQ(node->get_parameter(prefix + "limits_marker_publish_rate").as_double(), 1);
  rclcpp::NodeOptions options;
  options.parameter_overrides({rclcpp::Parameter(prefix + "publish_limits_marker", true),
    rclcpp::Parameter(prefix + "limits_marker_publish_rate", 5.0)});
  auto overrides = std::make_shared<rclcpp::Node>("limits_marker_overrides", options);
  declare_joint_limits_marker_parameters(overrides->get_node_parameters_interface(), prefix);
  EXPECT_TRUE(overrides->get_parameter(prefix + "publish_limits_marker").as_bool());
  EXPECT_DOUBLE_EQ(overrides->get_parameter(prefix + "limits_marker_publish_rate").as_double(), 5);
}

TEST_F(JointLimitsObserverTest, PublishesWithoutCollisionTaskAtConfiguredRateAndDeletes)
{
  auto backend = marker_backend(); auto task = marker_task(marker_parameters(true, 5), backend);
  auto node = std::make_shared<rclcpp::Node>("joint_limits_observer_test");
  bool received = false, cleared = false;
  size_t messages = 0;
  auto subscription = node->create_subscription<visualization_msgs::msg::MarkerArray>(
    std::string(node->get_fully_qualified_name()) + "/joint_limits/limits_markers", rclcpp::QoS(1),
    [&](visualization_msgs::msg::MarkerArray::SharedPtr array) {
      ++messages;
      ASSERT_EQ(array->markers.size(), 3u);
      const bool active = array->markers[0].action == Marker::ADD;
      if (active) {
        received = true; EXPECT_EQ(array->markers[0].type, Marker::CYLINDER);
        EXPECT_DOUBLE_EQ(array->markers[0].scale.x, .06); EXPECT_EQ(array->markers[0].header.frame_id, "base");
      } else {cleared = true;}
    });
  WholeBodyState state; state.joint_positions = Eigen::Vector3d(.05, .5, .5);
  TaskComputation output; task->prepare_computation(output);
  task->update_into(state, *backend, output); task->observe_command({}, 123);
  auto observer = std::make_unique<CapsuleObserver>(std::vector<std::shared_ptr<TaskBase>>{task},
    node->get_fully_qualified_name(), node->get_node_base_interface()->get_context());
  const auto wait_for = [&](const std::function<bool()> & condition) {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
      while (!condition() && std::chrono::steady_clock::now() < deadline) {
        rclcpp::spin_some(node); std::this_thread::sleep_for(std::chrono::milliseconds(5));
      }
    };
  wait_for([&] {return received;}); ASSERT_TRUE(received);
  // No further control updates: refresh still runs on the observer executor.
  const size_t before = messages;
  const auto end = std::chrono::steady_clock::now() + std::chrono::milliseconds(500);
  while (std::chrono::steady_clock::now() < end) {
    rclcpp::spin_some(node); std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_LE(messages - before, 4u);
  EXPECT_GE(messages - before, 1u);
  cleared = false; task->set_publish_limits_marker(false);
  wait_for([&] {return cleared;}); EXPECT_TRUE(cleared);
  received = false; cleared = false; task->set_publish_limits_marker(true);
  const auto fresh_deadline = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
  while (std::chrono::steady_clock::now() < fresh_deadline) {
    rclcpp::spin_some(node); std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  EXPECT_FALSE(received); // A pre-toggle snapshot must not be redisplayed.
  task->update_into(state, *backend, output); task->observe_command({}, 124);
  wait_for([&] {return received;}); ASSERT_TRUE(received);
  task->set_enabled(false); task->update_into(state, *backend, output); task->observe_command({}, 125);
  wait_for([&] {return cleared;}); EXPECT_TRUE(cleared);
  received = false; cleared = false; task->set_enabled(true);
  task->update_into(state, *backend, output); task->observe_command({}, 126);
  wait_for([&] {return received;}); ASSERT_TRUE(received);
  observer.reset(); wait_for([&] {return cleared;}); EXPECT_TRUE(cleared);
}
}  // namespace
}  // namespace task_priority_kinematic_control
