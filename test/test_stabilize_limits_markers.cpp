#include "task_priority_kinematic_control/core/capsule_observer.hpp"

#include <gtest/gtest.h>
#include <cmath>
#include <functional>
#include <limits>
#include <thread>
#include <stdexcept>

namespace task_priority_kinematic_control
{
namespace
{
using Marker = visualization_msgs::msg::Marker;
double radians(double degrees) {return degrees * std::acos(-1.0) / 180;}

StabilizeLimitsSnapshot snapshot()
{
  StabilizeLimitsSnapshot s;
  s.enabled = s.valid = s.visualization_enabled = true;
  s.lower.setConstant(radians(-20)); s.upper.setConstant(radians(20));
  s.alpha = radians(4); s.delta = radians(8);
  s.timestamp_ns = 123;
  return s;
}

class MarkerBackend : public KinematicsBackend
{
public:
  void configure(const WholeBodyModel &, const std::string &, const rclcpp::Logger &) override {}
  void update(const WholeBodyState &) override {}
  FrameState get_frame_state(const std::string &) const override {return {};}
  Eigen::Isometry3d get_relative_transform(const std::string &, const std::string &) const override
  {return Eigen::Isometry3d::Identity();}
  const std::vector<CollisionCapsule> & collision_capsules() const override {return capsules_;}
  std::string name() const override {return "tilt_marker_test";}
private:
  std::vector<CollisionCapsule> capsules_;
};

std::shared_ptr<StabilizeLimitsTask> task(bool publish = true, double rate = 2)
{
  auto model = std::make_shared<WholeBodyModel>();
  model->configure("world", "base", "left", "right", {}, {}, {0,1,2,3,4,5});
  auto result = std::make_shared<StabilizeLimitsTask>();
  result->configure("stabilize_limits", "task_priority_kinematic_control/StabilizeLimitsTask", {
    {"lower_limits", rclcpp::Parameter("lower_limits", std::vector<double>{radians(-20), radians(-20)})},
    {"upper_limits", rclcpp::Parameter("upper_limits", std::vector<double>{radians(20), radians(20)})},
    {"alpha", rclcpp::Parameter("alpha", radians(4))},
    {"delta", rclcpp::Parameter("delta", radians(8))},
    {"debug", rclcpp::Parameter("debug", publish)},
    {"publish_marker_rate", rclcpp::Parameter("publish_marker_rate", rate)}}, TaskContext{model});
  return result;
}
}  // namespace

TEST(StabilizeLimitsMarkers, CompactGaugesHaveCenteredArcsShortArrowsAndNoText)
{
  auto s = snapshot(); s.angles << radians(17), radians(-5);
  StabilizeLimitsMarkerRenderer renderer("base", "stabilize_limits");
  const auto & markers = renderer.update(s);
  ASSERT_EQ(markers.markers.size(), 14u);
  for (size_t i = 0; i < markers.markers.size(); ++i) {
    const auto & m = markers.markers[i];
    EXPECT_EQ(m.action, i % 7 == 6 ? Marker::DELETE : Marker::ADD);
    EXPECT_EQ(m.id, static_cast<int>(i % 7));
    EXPECT_EQ(m.ns, i < 7 ? "stabilize_limits/limits/roll" : "stabilize_limits/limits/pitch");
    EXPECT_EQ(m.header.frame_id, "base");
    EXPECT_TRUE(m.frame_locked);
    EXPECT_EQ(m.lifetime.sec, 2);
    EXPECT_NE(m.type, Marker::TEXT_VIEW_FACING);
    EXPECT_TRUE(m.text.empty());
    for (const auto & point : m.points) {
      EXPECT_LT(std::sqrt(point.x*point.x + point.y*point.y + point.z*point.z), .55);
    }
  }
  for (size_t axis = 0; axis < 2; ++axis) {
    const auto & arrow = markers.markers[axis*7 + 5];
    ASSERT_EQ(arrow.points.size(), 2u);
    EXPECT_EQ(arrow.type, Marker::ARROW);
    const auto & origin = arrow.points[0];
    EXPECT_DOUBLE_EQ(origin.x, 0);
    EXPECT_DOUBLE_EQ(origin.y, axis == 0 ? 0 : .22);
    EXPECT_DOUBLE_EQ(origin.z, axis == 0 ? -.15 : -.18);
    const auto distance = [&](const geometry_msgs::msg::Point & p) {
      return std::sqrt(std::pow(p.x-origin.x, 2) + std::pow(p.y-origin.y, 2) + std::pow(p.z-origin.z, 2));
    };
    EXPECT_NEAR(distance(arrow.points[1]), .16, 1e-12);
    const auto & arc = markers.markers[axis*7];
    ASSERT_EQ(arc.points.size(), 33u);
    for (const auto & point : arc.points) {EXPECT_NEAR(distance(point), .18, 1e-12);}
    EXPECT_EQ(markers.markers[axis*7 + 3].points.size(), 4u);
    EXPECT_EQ(markers.markers[axis*7 + 4].points.size(), 4u);
  }
}

TEST(StabilizeLimitsMarkers, ColorsDistinguishFreeNearActiveAndDisabledIndependently)
{
  auto s = snapshot();
  StabilizeLimitsMarkerRenderer renderer("base", "limits");
  const auto expect_color = [&](size_t axis, float red, float green, float blue) {
    const auto & markers = renderer.update(s);
    for (const size_t id : {0u, 1u, 2u, 5u}) {
      const auto & c = markers.markers[axis*7 + id].color;
      EXPECT_FLOAT_EQ(c.r, red); EXPECT_FLOAT_EQ(c.g, green); EXPECT_FLOAT_EQ(c.b, blue);
    }
  };
  expect_color(0, 0, 1, 0);
  s.angles.x() = radians(13); expect_color(0, 1, .55f, 0);
  s.active[0] = true; expect_color(0, 1, 0, 0);
  expect_color(1, 0, 1, 0);
  s.angles.x() = radians(21); expect_color(0, 1, 0, 0);
  s.enabled = false;
  expect_color(0, .5f, .5f, .5f); expect_color(1, .5f, .5f, .5f);
}

TEST(StabilizeLimitsMarkers, PerpendicularTicksMarkHardLimitsAndActivationWithFixedColors)
{
  auto s = snapshot();
  StabilizeLimitsMarkerRenderer renderer("base", "limits");
  for (const double angle : {0., 13., 18., 21.}) {
    s.angles.setConstant(radians(angle));
    s.active = {{angle >= 18, angle >= 18}};
    const auto & markers = renderer.update(s);
    for (size_t axis = 0; axis < 2; ++axis) {
      const auto & origin = markers.markers[axis*7 + 5].points[0];
      for (size_t id = 3; id <= 4; ++id) {
        const auto & marker = markers.markers[axis*7 + id];
        ASSERT_EQ(marker.points.size(), 4u);
        EXPECT_EQ(marker.type, Marker::LINE_LIST);
        EXPECT_FLOAT_EQ(marker.color.r, 1);
        EXPECT_FLOAT_EQ(marker.color.g, id == 3 ? 0 : .55f);
        EXPECT_FLOAT_EQ(marker.color.b, 0);
        for (size_t end = 0; end < 2; ++end) {
          const auto & p = marker.points[2*end];
          const double actual_angle = axis == 0 ?
            std::atan2(p.y-origin.y, -(p.z-origin.z)) :
            std::atan2(-(p.x-origin.x), -(p.z-origin.z));
          EXPECT_NEAR(actual_angle, radians((end == 0 ? -1 : 1) * (id == 3 ? 20 : 16)), 1e-12);
        }
      }
    }
  }
  s.enabled = false;
  for (const auto & marker : renderer.update(s).markers) {
    if (marker.action != Marker::ADD) {continue;}
    EXPECT_FLOAT_EQ(marker.color.r, .5f);
    EXPECT_FLOAT_EQ(marker.color.g, .5f);
    EXPECT_FLOAT_EQ(marker.color.b, .5f);
  }
}

TEST(StabilizeLimitsMarkers, RendererReusesAllBuffersEvenWhenGeometryChanges)
{
  auto s = snapshot();
  StabilizeLimitsMarkerRenderer renderer("base", "limits");
  const auto & markers = renderer.update(s);
  const auto * storage = markers.markers.data();
  std::array<const geometry_msgs::msg::Point *, 14> points{};
  std::array<const char *, 14> frames{}, namespaces{};
  for (size_t i = 0; i < 14; ++i) {
    points[i] = markers.markers[i].points.data();
    frames[i] = markers.markers[i].header.frame_id.data();
    namespaces[i] = markers.markers[i].ns.data();
  }
  for (int step = 0; step < 100; ++step) {
    s.angles.x() = radians(step % 20);
    if (step == 50) {s.upper.y() = radians(25); s.alpha = radians(3);}
    renderer.clear();
    const auto & next = renderer.update(s);
    EXPECT_EQ(next.markers.data(), storage);
    for (size_t i = 0; i < 14; ++i) {
      EXPECT_EQ(next.markers[i].points.data(), points[i]);
      EXPECT_EQ(next.markers[i].header.frame_id.data(), frames[i]);
      EXPECT_EQ(next.markers[i].ns.data(), namespaces[i]);
    }
  }
}

TEST(StabilizeLimitsMarkers, InvalidOrHiddenSnapshotsDeleteEveryMarker)
{
  StabilizeLimitsMarkerRenderer renderer("base", "limits");
  for (const bool visualization : {true, false}) {
    auto s = snapshot(); s.visualization_enabled = visualization; s.valid = false;
    for (const auto & m : renderer.update(s).markers) {EXPECT_EQ(m.action, Marker::DELETE);}
  }
  auto s = snapshot(); s.visualization_enabled = false;
  for (const auto & m : renderer.update(s).markers) {EXPECT_EQ(m.action, Marker::DELETE);}
  for (const auto & m : renderer.clear().markers) {EXPECT_EQ(m.action, Marker::DELETE);}
}

TEST(StabilizeLimitsMarkers, SnapshotsPreserveControlAndReportActualHysteresis)
{
  auto limits = task(false); MarkerBackend backend;
  WholeBodyState state; state.navigation_valid = true; state.base_rpy.x() = radians(19);
  auto before = limits->update(state, backend); limits->observe_command({}, 1000000000LL);
  EXPECT_EQ(limits->consume_snapshot(), nullptr);
  limits->set_debug(true);
  auto after = limits->update(state, backend); limits->observe_command({}, 2000000000LL);
  EXPECT_TRUE(before.jacobian.isApprox(after.jacobian));
  EXPECT_TRUE(before.desired_velocity.isApprox(after.desired_velocity));
  const auto * captured = limits->consume_snapshot(); ASSERT_NE(captured, nullptr);
  EXPECT_TRUE(captured->active[0]); EXPECT_FALSE(captured->active[1]);
  EXPECT_TRUE(captured->valid); EXPECT_EQ(captured->timestamp_ns, 2000000000LL);
  state.base_rpy.x() = radians(17);
  limits->update(state, backend); limits->observe_command({}, 3000000000LL);
  captured = limits->consume_snapshot(); ASSERT_NE(captured, nullptr);
  EXPECT_TRUE(captured->active[0]);  // Still active due to hysteresis.
  limits->set_enabled(false); limits->update(state, backend); limits->observe_command({}, 4000000000LL);
  captured = limits->consume_snapshot(); ASSERT_NE(captured, nullptr);
  EXPECT_FALSE(captured->enabled); EXPECT_FALSE(captured->active[0]); EXPECT_TRUE(captured->valid);
  state.navigation_valid = false; limits->update(state, backend); limits->observe_command({}, 5000000000LL);
  captured = limits->consume_snapshot(); ASSERT_NE(captured, nullptr); EXPECT_FALSE(captured->valid);
}

TEST(StabilizeLimitsMarkers, SamplesAtConfiguredRateAndRefreshesAfterDebugToggleOrClockRewind)
{
  auto limits = task(); MarkerBackend backend;
  WholeBodyState state; state.navigation_valid = true;
  TaskComputation computation; limits->prepare_computation(computation);
  int samples = 0;
  for (int64_t stamp = 0; stamp <= 1000000000LL; stamp += 10000000LL) {
    limits->update_into(state, backend, computation); limits->observe_command({}, stamp);
    if (const auto * sample = limits->consume_snapshot()) {
      EXPECT_EQ(sample->timestamp_ns % 500000000LL, 0);
      ++samples;
    }
  }
  EXPECT_EQ(samples, 3);
  limits->set_debug(false);
  limits->update_into(state, backend, computation); limits->observe_command({}, 1100000000LL);
  EXPECT_EQ(limits->consume_snapshot(), nullptr);
  limits->set_debug(true);
  limits->update_into(state, backend, computation); limits->observe_command({}, 1110000000LL);
  ASSERT_NE(limits->consume_snapshot(), nullptr);
  limits->observe_command({}, 10);  // ROS time rewind forces a fresh sample.
  const auto * sample = limits->consume_snapshot(); ASSERT_NE(sample, nullptr);
  EXPECT_EQ(sample->timestamp_ns, 10);
}

TEST(StabilizeLimitsMarkers, VisualizationParametersValidateAtomically)
{
  for (double rate : {0., -1., std::numeric_limits<double>::infinity(), std::numeric_limits<double>::quiet_NaN()}) {
    EXPECT_THROW(task(false, rate), std::invalid_argument);
  }
  auto limits = task(false);
  const std::vector<std::shared_ptr<TaskBase>> tasks{limits};
  const std::string prefix = "tasks.stabilize_limits.";
  EXPECT_FALSE(validate_joint_limits_marker_parameter_updates(
    {rclcpp::Parameter(prefix + "debug", "true")}, tasks).successful);
  EXPECT_FALSE(validate_joint_limits_marker_parameter_updates(
    {rclcpp::Parameter(prefix + "debug", true),
      rclcpp::Parameter(prefix + "publish_marker_rate", 2.)}, tasks).successful);
  EXPECT_FALSE(limits->debug());
  EXPECT_TRUE(validate_joint_limits_marker_parameter_updates(
    {rclcpp::Parameter(prefix + "debug", true)}, tasks, false).successful);
  EXPECT_FALSE(limits->debug());
  EXPECT_TRUE(validate_joint_limits_marker_parameter_updates(
    {rclcpp::Parameter(prefix + "debug", true)}, tasks).successful);
  EXPECT_TRUE(limits->debug());
}

class StabilizeLimitsObserverTest : public ::testing::Test
{
protected:
  static void SetUpTestSuite() {rclcpp::init(0, nullptr);}
  static void TearDownTestSuite() {rclcpp::shutdown();}
};

TEST_F(StabilizeLimitsObserverTest, PublishesAloneAndDeletesHiddenStaleAndShutdownMarkers)
{
  auto limits = task(); MarkerBackend backend;
  auto node = std::make_shared<rclcpp::Node>("stabilize_limits_observer_test");
  int added = 0, deleted = 0;
  auto sub = node->create_subscription<visualization_msgs::msg::MarkerArray>(
    std::string(node->get_fully_qualified_name()) + "/stabilize_limits/limits_markers", 1,
    [&](visualization_msgs::msg::MarkerArray::SharedPtr array) {
      ASSERT_EQ(array->markers.size(), 14u);
      if (array->markers[0].action == Marker::ADD) {++added;} else {++deleted;}
    });
  auto observer = std::make_unique<CapsuleObserver>(std::vector<std::shared_ptr<TaskBase>>{limits},
    node->get_fully_qualified_name(), node->get_node_base_interface()->get_context());
  WholeBodyState state; state.navigation_valid = true; state.base_rpy.x() = radians(17);
  const auto capture = [&] {limits->update(state, backend); limits->observe_command({}, node->now().nanoseconds());};
  const auto wait_for = [&](const std::function<bool()> & condition, bool produce) {
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
    while (!condition() && std::chrono::steady_clock::now() < deadline) {
      if (produce) {capture();}
      rclcpp::spin_some(node); std::this_thread::sleep_for(std::chrono::milliseconds(5));
    }
  };
  wait_for([&] {return added > 0;}, true); ASSERT_GT(added, 0);
  wait_for([&] {return deleted > 0;}, false); ASSERT_GT(deleted, 0);  // Stale navigation/control data.
  const int old_added = added;
  wait_for([&] {return added > old_added;}, true); ASSERT_GT(added, old_added);
  const int old_deleted = deleted;
  limits->set_debug(false);
  wait_for([&] {return deleted > old_deleted;}, false); ASSERT_GT(deleted, old_deleted);
  limits->set_debug(true);
  const int hidden_added = added;
  const auto until = std::chrono::steady_clock::now() + std::chrono::milliseconds(250);
  while (std::chrono::steady_clock::now() < until) {rclcpp::spin_some(node);}
  EXPECT_EQ(added, hidden_added);  // Old snapshots must not reappear after a toggle.
  wait_for([&] {return added > hidden_added;}, true); ASSERT_GT(added, hidden_added);
  const int before_shutdown = deleted;
  observer.reset();
  wait_for([&] {return deleted > before_shutdown;}, false); EXPECT_GT(deleted, before_shutdown);
}

}  // namespace task_priority_kinematic_control
