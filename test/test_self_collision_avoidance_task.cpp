#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include "task_priority_kinematic_control/core/task_manager.hpp"
#include "task_priority_kinematic_control/geometry/capsule_geometry.hpp"
#include "task_priority_kinematic_control/geometry/plane_geometry.hpp"
#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"
#include "task_priority_kinematic_control/core/capsule_observer.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>
#include <algorithm>
#include <sstream>
#include <thread>
#include <chrono>
#include "task_priority_kinematic_control/kinematics/kdl_kinematics_backend.hpp"

namespace task_priority_kinematic_control
{
namespace
{
using V = Eigen::Vector3d;
using Params = std::map<std::string,rclcpp::Parameter>;
class Backend : public KinematicsBackend
{
public:
  void configure(const WholeBodyModel &,const std::string &,const rclcpp::Logger &) override {}
  void update(const WholeBodyState &) override {}
  FrameState get_frame_state(const std::string & name) const override {return frames.at(name);}
  bool read_frame_state(const std::string & name, FrameState & out) const override
  {
    const auto it = frames.find(name);
    if (it == frames.end()) {return false;}
    out.pose = it->second.pose; out.jacobian = it->second.jacobian; return true;
  }
  Eigen::Isometry3d get_relative_transform(const std::string & a,const std::string & b) const override
  {return frames.at(a).pose.inverse()*frames.at(b).pose;}
  const std::vector<CollisionCapsule> & collision_capsules() const override {return old_capsules;}
  std::string name() const override {return "mock";}
  std::map<std::string,FrameState> frames;
  std::vector<CollisionCapsule> old_capsules;
};
class CertifiedBackend : public Backend
{
public:
  bool frame_joint_dependencies(const std::string & frame, std::vector<size_t> & indices) const override
  {
    ++dependency_calls;
    indices.clear();
    const auto it = dependencies.find(frame);
    if (it == dependencies.end()) {return false;}
    indices = it->second;
    return true;
  }
  std::map<std::string, std::vector<size_t>> dependencies;
  mutable size_t dependency_calls = 0;
};
std::shared_ptr<WholeBodyModel> model()
{
  auto m=std::make_shared<WholeBodyModel>();
  m->configure("world","base","B","D",{"l0","l1","l2","l3"},{"r0","r1","r2","r3"},{0,1,2,3,4,5});
  return m;
}
void scalar(Params & p,const std::string & key,double value) {p[key]=rclcpp::Parameter(key,value);}
void capsule(Params & p,const std::string & name,const std::string & a,const std::string & b,double radius=0.04)
{
  const auto key="capsules."+name+".";
  p[key+"start_frame"]=rclcpp::Parameter(key+"start_frame",a);
  p[key+"end_frame"]=rclcpp::Parameter(key+"end_frame",b);
  scalar(p,key+"radius",radius);
}
Params parameters()
{
  Params p;
  p["plugin"]=rclcpp::Parameter("plugin","task_priority_kinematic_control/SelfCollisionAvoidanceTask");
  p["enabled"]=rclcpp::Parameter("enabled",true);
  scalar(p,"priority",0.0); scalar(p,"safe_distance",0.05); scalar(p,"activation_distance",0.12);
  scalar(p,"gain_scalar",1.0); scalar(p,"max_repulsive_velocity",0.08);
  p["capsule_names"]=rclcpp::Parameter("capsule_names",std::vector<std::string>{"a","b"});
  capsule(p,"a","A","B"); capsule(p,"b","C","D");
  return p;
}
Params plane_parameters()
{
  auto p = parameters();
  p["ignored_collision_pairs"] = rclcpp::Parameter("ignored_collision_pairs", std::vector<std::string>{"a,b"});
  p["plane_names"] = rclcpp::Parameter("plane_names", std::vector<std::string>{"guard"});
  p["planes.guard.normal_axis"] = rclcpp::Parameter("planes.guard.normal_axis", "z");
  scalar(p, "planes.guard.position", -0.1);
  p["planes.guard.allowed_side"] = rclcpp::Parameter("planes.guard.allowed_side", "positive");
  p["planes.guard.bounds_min"] = rclcpp::Parameter("planes.guard.bounds_min", std::vector<double>{-0.5, -0.5});
  p["planes.guard.bounds_max"] = rclcpp::Parameter("planes.guard.bounds_max", std::vector<double>{0.5, 0.5});
  p["planes.guard.checked_capsules"] = rclcpp::Parameter("planes.guard.checked_capsules", std::vector<std::string>{"a"});
  return p;
}
std::shared_ptr<Backend> backend(const Eigen::VectorXd & q=Eigen::VectorXd::Zero(14),
  const Eigen::Isometry3d & pose=Eigen::Isometry3d::Identity())
{
  auto b=std::make_shared<Backend>();
  auto add=[&](const std::string & name,V p,int col,double weight) {
      FrameState frame; frame.pose=pose;
      if(col>=0) {p.z()+=weight*q(col);}
      const V r=pose.rotation()*p;
      frame.pose.translation()+=r;
      frame.jacobian=Eigen::MatrixXd::Zero(6,14);
      frame.jacobian.topLeftCorner(3,3).setIdentity();
      frame.jacobian.block<3,3>(0,3)=-skew(r);
      frame.jacobian.block<3,3>(3,3).setIdentity();
      if(col>=0) {frame.jacobian.col(col).head<3>()=pose.rotation()*V(0,0,weight);}
      b->frames[name]=frame;
    };
  add("base",V::Zero(),-1,0);
  add("A",{-0.1,0,0},6,1.0); add("B",{0.1,0,0},7,0.4);
  add("C",{0,-0.1,0.1},10,0.6); add("D",{0,0.1,0.1},11,1.0);
  return b;
}
void configure(SelfCollisionAvoidanceTask & task,const Params & p,const std::shared_ptr<Backend> & b)
{task.configure("self_collision_avoidance","task_priority_kinematic_control/SelfCollisionAvoidanceTask",p,TaskContext{model(),b});}

TEST(SelfCollisionPlanes, ActivationContactCrossingSaturationAndRecovery)
{
  auto b = backend(); auto p = plane_parameters(); SelfCollisionAvoidanceTask task; configure(task, p, b);
  ASSERT_EQ(task.planes().size(), 1u); EXPECT_EQ(task.planes()[0].reference_frame, "base");
  EXPECT_EQ(task.checked_pairs().size(), 0u); EXPECT_EQ(task.checked_plane_pairs().size(), 1u);
  auto out = task.update({}, *b);
  ASSERT_EQ(out.jacobian.rows(), 1); EXPECT_TRUE(out.active); EXPECT_FALSE(out.stop_arm_motion);
  EXPECT_NEAR(out.desired_velocity(0), 0.07, 1e-12); EXPECT_DOUBLE_EQ(out.error(0), 0);
  EXPECT_NEAR(out.jacobian(0, 6), 1, 1e-12);
  EXPECT_EQ(task.collision_pair_name(0), "a,plane:guard");
  EXPECT_NE(task.build_status().status_message.find("closest_pair=a,plane:guard"), std::string::npos);
  for (const double z : {-0.10, -0.20}) {
    b->frames["A"].pose.translation().z() = z;
    b->frames["B"].pose.translation().z() = z;
    out = task.update({}, *b); EXPECT_TRUE(out.active); EXPECT_FALSE(out.stop_arm_motion);
    EXPECT_NEAR(out.desired_velocity(0), 0.08, 1e-12); EXPECT_NEAR(out.jacobian(0, 6), 1, 1e-12);
  }
  for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().z() = 0.2;}
  out = task.update({}, *b); EXPECT_FALSE(out.active); EXPECT_FALSE(out.stop_arm_motion);
  EXPECT_EQ(out.jacobian.rows(), 1); EXPECT_DOUBLE_EQ(out.jacobian.norm(), 0);
  task.set_enabled(false); out = task.update({}, *b); EXPECT_FALSE(out.active); EXPECT_FALSE(out.stop_arm_motion);
}

TEST(SelfCollisionPlanes, FiniteExtentSelectionMultiplePlanesAndOppositeSide)
{
  auto b = backend(); auto p = plane_parameters();
  for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().y() = 0.55;}
  SelfCollisionAvoidanceTask task; configure(task, p, b); EXPECT_FALSE(task.update({}, *b).active);
  for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().y() = 0.52;}
  EXPECT_TRUE(task.update({}, *b).active); // Radius extends the finite influence region.
  p.erase("planes.guard.checked_capsules");
  p["plane_names"] = rclcpp::Parameter("plane_names", std::vector<std::string>{"guard", "second"});
  const auto first = p;
  for (const auto & parameter : first) {
    if (parameter.first.rfind("planes.guard.", 0) == 0) {
      const auto key = "planes.second." + parameter.first.substr(13);
      p[key] = rclcpp::Parameter(key, parameter.second.get_parameter_value());
    }
  }
  p["planes.second.allowed_side"] = rclcpp::Parameter("planes.second.allowed_side", "negative");
  configure(task, p, b); const auto out = task.update({}, *b);
  EXPECT_EQ(task.checked_plane_pairs().size(), 4u); ASSERT_EQ(out.jacobian.rows(), 4);
  EXPECT_GT(out.jacobian(0, 6), 0); EXPECT_LT(out.jacobian(2, 6), 0);
  EXPECT_FALSE(out.stop_arm_motion);
}

// Moving reference has independent angular (column 10) and translational (11) DoFs.
std::shared_ptr<Backend> plane_backend(
  const Eigen::VectorXd & q, const Eigen::Isometry3d & common = Eigen::Isometry3d::Identity())
{
  auto b = backend(q, common);
  const V axis = V(1, 2, 3).normalized(), velocity(0.1, -0.2, 0.3);
  Eigen::Isometry3d local = Eigen::Isometry3d::Identity();
  local.linear() = Eigen::AngleAxisd(0.4 + q(10), axis).toRotationMatrix();
  local.translation() = V(0.01, -0.02, 0.03) + q(11) * velocity;
  FrameState reference = b->frames.at("base"); reference.pose = common * local;
  const V r = common.rotation() * local.translation();
  reference.jacobian.block<3, 3>(0, 3) = -skew(r);
  reference.jacobian.col(10).tail<3>() = common.rotation() * axis;
  reference.jacobian.col(11).head<3>() = common.rotation() * velocity;
  b->frames["reference"] = reference;
  return b;
}

TEST(SelfCollisionPlanes, RotatedMovingReferenceJacobiansAndBaseInvariance)
{
  auto p = plane_parameters(); scalar(p, "activation_distance", 10);
  p["planes.guard.reference_frame"] = rclcpp::Parameter("planes.guard.reference_frame", "reference");
  Eigen::VectorXd q = Eigen::VectorXd::Zero(14); q(6) = 0.01; q(7) = 0.03;
  auto b = plane_backend(q); SelfCollisionAvoidanceTask task; configure(task, p, b);
  const auto out = task.update({}, *b); ASSERT_TRUE(out.active); ASSERT_FALSE(out.stop_arm_motion);
  auto clearance = [&](const Eigen::VectorXd & positions) {
      const auto perturbed = plane_backend(positions);
      const auto inverse = perturbed->frames.at("reference").pose.inverse();
      return capsule_plane_distance(inverse * perturbed->frames.at("A").pose.translation(),
        inverse * perturbed->frames.at("B").pose.translation(), .04, 2, 1, -.1,
        Eigen::Vector2d(-.5, -.5), Eigen::Vector2d(.5, .5)).clearance;
    };
  for (int col : {6, 7, 10, 11}) {
    auto plus = q, minus = q; plus(col) += 1e-6; minus(col) -= 1e-6;
    EXPECT_NEAR(out.jacobian(0, col), (clearance(plus) - clearance(minus)) / 2e-6, 1e-8) << col;
  }
  EXPECT_GT(std::abs(out.jacobian(0, 10)), 1e-3);
  EXPECT_GT(std::abs(out.jacobian(0, 11)), 1e-3);
  Eigen::Isometry3d common = Eigen::Isometry3d::Identity();
  common.linear() = Eigen::AngleAxisd(.7, V(1, 2, 1).normalized()).toRotationMatrix();
  common.translation() = V(2, 3, 4);
  auto moved = plane_backend(q, common); SelfCollisionAvoidanceTask other; configure(other, p, moved);
  const auto transformed = other.update({}, *moved);
  EXPECT_LT((out.jacobian - transformed.jacobian).norm(), 1e-12);
  EXPECT_LT((out.desired_velocity - transformed.desired_velocity).norm(), 1e-12);
  EXPECT_DOUBLE_EQ(out.jacobian.leftCols(6).norm(), 0);
  WholeBodyCommand command; command.generalized_velocity = Eigen::VectorXd::Zero(14);
  task.observe_command(command, 123); other.observe_command(command, 123);
  const auto * snapshot = task.consume_snapshot(); const auto * moved_snapshot = other.consume_snapshot();
  ASSERT_NE(snapshot, nullptr); ASSERT_NE(moved_snapshot, nullptr);
  ASSERT_TRUE(snapshot->planes[0].valid);
  const V expected = b->frames.at("reference").pose * V(-.5, -.5, -.1);
  EXPECT_LT((snapshot->planes[0].corners[0] - expected).norm(), 1e-12);
  for (size_t i = 0; i < 4; ++i) {
    EXPECT_LT((snapshot->planes[0].corners[i] - moved_snapshot->planes[0].corners[i]).norm(), 1e-12);
  }
  EXPECT_NEAR(snapshot->metrics.max_velocity_deficit, out.desired_velocity(0), 1e-12);
  // Subsequent angular motion updates the rectangle, not just its collision Jacobian.
  q(10) += .2; auto rotated = plane_backend(q); task.update({}, *rotated); task.observe_command(command, 124);
  const auto * next = task.consume_snapshot(); ASSERT_NE(next, nullptr);
  const V next_expected = rotated->frames.at("reference").pose * V(-.5, -.5, -.1);
  EXPECT_LT((next->planes[0].corners[0] - next_expected).norm(), 1e-12);
}

TEST(SelfCollisionPlanes, ClippedTaskJacobianIncludesTangentialEndpointMotion)
{
  auto p = plane_parameters();
  p["planes.guard.normal_axis"] = rclcpp::Parameter("planes.guard.normal_axis", "x");
  scalar(p, "planes.guard.position", 0.0);
  p["planes.guard.bounds_min"] = rclcpp::Parameter("planes.guard.bounds_min", std::vector<double>{-.2, -.3});
  p["planes.guard.bounds_max"] = rclcpp::Parameter("planes.guard.bounds_max", std::vector<double>{.2, .3});
  auto b = std::make_shared<CertifiedBackend>(); b->frames = backend()->frames;
  b->dependencies = {{"base", {}}, {"A", {0}}, {"B", {1}}, {"C", {4}}, {"D", {5}}};
  b->frames["A"].pose.translation() = V(.1, -.5, 0);
  b->frames["B"].pose.translation() = V(.2, .5, 0);
  for (const auto & name : {"A", "B"}) {
    b->frames[name].jacobian.block<3, 3>(0, 3) = -skew(b->frames[name].pose.translation());
  }
  b->frames["A"].jacobian.col(6).head<3>() = V(0, 1, 0);
  b->frames["B"].jacobian.col(7).head<3>() = V(0, 1, 0);
  SelfCollisionAvoidanceTask task; configure(task, p, b); const auto out = task.update({}, *b);
  ASSERT_TRUE(out.active);
  for (int col : {6, 7}) {
    const V a = b->frames["A"].pose.translation(), end = b->frames["B"].pose.translation();
    V da = V::Zero(), db = V::Zero();
    if (col == 6) {da.y() = 1e-6;} else {db.y() = 1e-6;}
    const auto plus = capsule_plane_distance(a + da, end + db, .04, 0, 1, 0, {-.2, -.3}, {.2, .3});
    const auto minus = capsule_plane_distance(a - da, end - db, .04, 0, 1, 0, {-.2, -.3}, {.2, .3});
    EXPECT_NEAR(out.jacobian(0, col), (plus.clearance - minus.clearance) / 2e-6, 1e-9);
  }
}

TEST(SelfCollisionPlanes, InvalidConfigurationAndRuntimeReferenceFailure)
{
  auto b = backend();
  for (const auto & entry : std::vector<std::pair<std::string, std::string>>{
      {"normal_axis", "w"}, {"allowed_side", "both"}, {"reference_frame", "missing"}}) {
    auto p = plane_parameters(); const auto key = "planes.guard." + entry.first;
    p[key] = rclcpp::Parameter(key, entry.second); SelfCollisionAvoidanceTask task;
    EXPECT_THROW(configure(task, p, b), std::invalid_argument);
  }
  for (const auto & values : std::vector<std::vector<double>>{{}, {0}, {.5, .5},
      {std::numeric_limits<double>::infinity(), 0}}) {
    auto p = plane_parameters();
    p["planes.guard.bounds_min"] = rclcpp::Parameter("planes.guard.bounds_min", values);
    SelfCollisionAvoidanceTask task; EXPECT_THROW(configure(task, p, b), std::invalid_argument);
  }
  for (const auto & values : std::vector<std::vector<std::string>>{{"unknown"}, {"a", "a"}}) {
    auto p = plane_parameters();
    p["planes.guard.checked_capsules"] = rclcpp::Parameter("planes.guard.checked_capsules", values);
    SelfCollisionAvoidanceTask task; EXPECT_THROW(configure(task, p, b), std::invalid_argument);
  }
  for (const auto & values : std::vector<std::vector<std::string>>{{"guard", "guard"}, {"bad-name"}}) {
    auto p = plane_parameters(); p["plane_names"] = rclcpp::Parameter("plane_names", values);
    SelfCollisionAvoidanceTask task; EXPECT_THROW(configure(task, p, b), std::invalid_argument);
  }
  auto p = plane_parameters(); scalar(p, "planes.guard.position", std::numeric_limits<double>::quiet_NaN());
  SelfCollisionAvoidanceTask task; EXPECT_THROW(configure(task, p, b), std::invalid_argument);
  p = plane_parameters(); scalar(p, "planes.guard.typo", 0); EXPECT_THROW(configure(task, p, b), std::invalid_argument);
  p = plane_parameters(); p.erase("planes.guard.position"); EXPECT_THROW(configure(task, p, b), std::invalid_argument);
  p = plane_parameters(); p["planes.guard.reference_frame"] = rclcpp::Parameter("planes.guard.reference_frame", "C");
  configure(task, p, b); b->frames["C"].jacobian(0, 10) = std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(task.update({}, *b).stop_arm_motion);
  WholeBodyCommand command; command.generalized_velocity = Eigen::VectorXd::Zero(14); task.observe_command(command, 1);
  const auto * snapshot = task.consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_FALSE(snapshot->planes[0].valid);
  b = backend(); EXPECT_FALSE(task.update({}, *b).stop_arm_motion);
  b->frames.erase("C"); EXPECT_TRUE(task.update({}, *b).stop_arm_motion);
}
TEST(SelfCollisionTask, SurfaceClearanceActivationAndFixedDimensions)
{
  auto b=backend(); SelfCollisionAvoidanceTask task; configure(task,parameters(),b);
  TaskComputation out; task.prepare_computation(out);
  task.update_into({},*b,out);
  EXPECT_TRUE(out.active); EXPECT_FALSE(out.stop_arm_motion);
  ASSERT_EQ(out.jacobian.rows(),1); EXPECT_EQ(out.jacobian.cols(),14);
  EXPECT_NEAR(out.desired_velocity(0),0.08,1e-12); EXPECT_NEAR(out.error(0),0.03,1e-12);
  EXPECT_DOUBLE_EQ(out.jacobian.leftCols(6).norm(),0);
  for(const auto & name : {"C","D"}) {b->frames[name].pose.translation().z()=0.3;}
  task.update_into({},*b,out);
  EXPECT_FALSE(out.active); EXPECT_EQ(out.jacobian.rows(),1); EXPECT_EQ(out.jacobian.cols(),14);
  EXPECT_DOUBLE_EQ(out.jacobian.norm(),0); EXPECT_DOUBLE_EQ(out.desired_velocity.norm(),0);
  // Repulsion starts before safe_distance is violated.
  for(const auto & name : {"C","D"}) {b->frames[name].pose.translation().z()=0.18;}
  task.update_into({},*b,out);
  EXPECT_TRUE(out.active); EXPECT_NEAR(out.desired_velocity(0),0.03,1e-12); EXPECT_DOUBLE_EQ(out.error(0),0);
}
TEST(SelfCollisionTask, ExactHysteresisThresholdsForCapsulesAndPlanes)
{
  // Binary-exact geometry makes equality at both thresholds unambiguous.
  for (const bool plane : {false, true}) {
    SCOPED_TRACE(plane ? "plane" : "capsules");
    auto b = backend(); auto p = plane ? plane_parameters() : parameters();
    scalar(p, "activation_distance", .125); scalar(p, "release_distance", .25);
    constexpr double tolerance = 1.0 / 1024;
    scalar(p, "eps", tolerance);
    scalar(p, "capsules.a.radius", .03125); scalar(p, "capsules.b.radius", .03125);
    if (plane) {scalar(p, "planes.guard.position", 0);}
    const auto set_clearance = [&](double clearance) {
        for (const auto & name : (plane ? std::vector<std::string>{"A", "B"} :
          std::vector<std::string>{"C", "D"})) {
          b->frames[name].pose.translation().z() = clearance + (plane ? .03125 : .0625);
        }
      };
    SelfCollisionAvoidanceTask task; configure(task, p, b);
    set_clearance(.125); EXPECT_FALSE(task.update({}, *b).active);
    set_clearance(.125 - tolerance); EXPECT_TRUE(task.update({}, *b).active);
    set_clearance(.1875); auto out = task.update({}, *b);
    EXPECT_TRUE(out.active); EXPECT_NEAR(out.desired_velocity(0), .0625, 1e-12);
    WholeBodyCommand command; command.generalized_velocity = Eigen::VectorXd::Zero(14);
    task.observe_command(command, 1); const auto * snapshot = task.consume_snapshot(); ASSERT_NE(snapshot, nullptr);
    EXPECT_TRUE(snapshot->capsules[0].highlighted);
    if (!plane) {EXPECT_TRUE(snapshot->capsules[1].highlighted);}
    set_clearance(.25 - 1.5 * tolerance); EXPECT_TRUE(task.update({}, *b).active);
    set_clearance(.25 - tolerance); out = task.update({}, *b);
    EXPECT_FALSE(out.active); EXPECT_DOUBLE_EQ(out.jacobian.norm(), 0);
    EXPECT_DOUBLE_EQ(out.desired_velocity.norm(), 0); EXPECT_DOUBLE_EQ(out.error.norm(), 0);
    task.observe_command(command, 2); snapshot = task.consume_snapshot(); ASSERT_NE(snapshot, nullptr);
    EXPECT_FALSE(snapshot->capsules[0].highlighted); EXPECT_FALSE(snapshot->capsules[1].highlighted);
    set_clearance(.1875); EXPECT_FALSE(task.update({}, *b).active);
    set_clearance(.125 - tolerance); EXPECT_TRUE(task.update({}, *b).active);
  }
}

TEST(SelfCollisionTask, ReleaseOnePairRestoresLowerTaskWhileAnotherPairStaysActive)
{
  auto b = backend(); auto p = plane_parameters();
  scalar(p, "planes.guard.position", 0);
  p.erase("planes.guard.checked_capsules");
  for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().z() = .05;}
  SelfCollisionAvoidanceTask task; configure(task, p, b);
  auto avoidance = task.update({}, *b); ASSERT_TRUE(avoidance.active);
  ASSERT_EQ(avoidance.jacobian.rows(), 2);
  EXPECT_NE(task.build_status().status_message.find("active_joint_count=2"), std::string::npos);
  TaskComputation lower;
  lower.active = true; lower.jacobian = Eigen::MatrixXd::Zero(1, 14); lower.jacobian(0, 6) = 1;
  lower.desired_velocity = Eigen::VectorXd::Constant(1, .2);
  HierarchySolver solver; solver.configure(14); solver.set_method(SolverMethod::kPinv);
  EXPECT_NEAR(solver.solve({avoidance, lower}).generalized_velocity(6), .08, 1e-12);
  for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().z() = .2;}
  avoidance = task.update({}, *b); ASSERT_TRUE(avoidance.active);
  EXPECT_DOUBLE_EQ(avoidance.jacobian.row(0).norm(), 0);
  EXPECT_DOUBLE_EQ(avoidance.desired_velocity(0), 0); EXPECT_DOUBLE_EQ(avoidance.error(0), 0);
  EXPECT_GT(avoidance.jacobian.row(1).norm(), 0);
  EXPECT_NEAR(solver.solve({avoidance, lower}).generalized_velocity(6), .2, 1e-12);
  EXPECT_NE(task.build_status().status_message.find("active_pairs=1"), std::string::npos);
  EXPECT_NE(task.build_status().status_message.find("active_joint_count=1"), std::string::npos);
}

TEST(SelfCollisionTask, StatesResetOnEnableResetReconfigureAndInvalidGeometry)
{
  auto b = backend(); auto p = plane_parameters(); SelfCollisionAvoidanceTask task; configure(task, p, b);
  const auto band = [&] {
      for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().z() = .065;}
    }; // .125 clearance: above entry, below release.
  const auto enter = [&] {
      for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().z() = 0;}
      EXPECT_TRUE(task.update({}, *b).active); band(); EXPECT_TRUE(task.update({}, *b).active);
    };
  enter(); task.reset(); EXPECT_FALSE(task.update({}, *b).active);
  enter(); task.set_enabled(false); EXPECT_FALSE(task.update({}, *b).active);
  task.set_enabled(true); EXPECT_FALSE(task.update({}, *b).active);
  enter(); configure(task, p, b); EXPECT_FALSE(task.update({}, *b).active);
  enter(); b->frames["A"].jacobian(0, 6) = std::numeric_limits<double>::quiet_NaN();
  const auto fault = task.update({}, *b); EXPECT_TRUE(fault.stop_arm_motion);
  EXPECT_DOUBLE_EQ(fault.jacobian.norm(), 0); EXPECT_DOUBLE_EQ(fault.desired_velocity.norm(), 0);
  b->frames["A"].jacobian(0, 6) = 0;
  auto recovered = task.update({}, *b); EXPECT_FALSE(recovered.stop_arm_motion); EXPECT_FALSE(recovered.active);
  enter();
}

TEST(SelfCollisionTask, MixedRowsRetainOrderAndReleaseIndependently)
{
  auto b = backend(); auto p = plane_parameters(); p.erase("ignored_collision_pairs");
  SelfCollisionAvoidanceTask task; configure(task, p, b);
  auto out = task.update({}, *b); ASSERT_EQ(out.jacobian.rows(), 2); ASSERT_TRUE(out.active);
  EXPECT_EQ(task.collision_pair_name(0), "a,b"); EXPECT_EQ(task.collision_pair_name(1), "a,plane:guard");
  EXPECT_GT(out.desired_velocity(0), 0); EXPECT_GT(out.desired_velocity(1), 0);
  for (const auto & name : {"C", "D"}) {b->frames[name].pose.translation().z() = .25;}
  out = task.update({}, *b); EXPECT_TRUE(out.active);
  EXPECT_DOUBLE_EQ(out.jacobian.row(0).norm(), 0);
  EXPECT_DOUBLE_EQ(out.desired_velocity(0), 0); EXPECT_DOUBLE_EQ(out.error(0), 0);
  EXPECT_GT(out.jacobian.row(1).norm(), 0); EXPECT_GT(out.desired_velocity(1), 0);
}

TEST(SelfCollisionPlanes, LeavingFiniteRegionClearsLatchAndMarkersFollowHysteresis)
{
  auto b = backend(); SelfCollisionAvoidanceTask task; configure(task, plane_parameters(), b);
  EXPECT_TRUE(task.update({}, *b).active);
  for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().z() = .065;}
  EXPECT_TRUE(task.update({}, *b).active);
  WholeBodyCommand command; command.generalized_velocity = Eigen::VectorXd::Zero(14);
  task.observe_command(command, 1); const auto * snapshot = task.consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_TRUE(snapshot->capsules[0].highlighted); EXPECT_TRUE(snapshot->planes[0].highlighted);
  const auto red = capsule_markers(task.capsules(), *snapshot, "base", 1e-9, task.planes());
  EXPECT_FLOAT_EQ(red.markers[0].color.r, 1); EXPECT_FLOAT_EQ(red.markers[0].color.b, 0);
  EXPECT_FLOAT_EQ(red.markers.back().color.r, 1); EXPECT_FLOAT_EQ(red.markers.back().color.g, 0);
  for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().y() = .6;}
  auto out = task.update({}, *b); EXPECT_FALSE(out.active); EXPECT_DOUBLE_EQ(out.jacobian.norm(), 0);
  for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().y() = 0;}
  EXPECT_FALSE(task.update({}, *b).active); // Re-entering the footprint in the hysteresis band.
  task.observe_command(command, 2); snapshot = task.consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_FALSE(snapshot->capsules[0].highlighted); EXPECT_FALSE(snapshot->planes[0].highlighted);
  const auto clear = capsule_markers(task.capsules(), *snapshot, "base", 1e-9, task.planes());
  EXPECT_GT(clear.markers.back().color.g, 0);
  // Releasing by distance also removes the red marker while still inside the footprint.
  for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().z() = 0;}
  EXPECT_TRUE(task.update({}, *b).active);
  for (const auto & name : {"A", "B"}) {b->frames[name].pose.translation().z() = .07;}
  EXPECT_FALSE(task.update({}, *b).active);
  task.observe_command(command, 3); snapshot = task.consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_FALSE(snapshot->planes[0].highlighted);
}

TEST(SelfCollisionTask, CertifiedMasksRemoveUnrelatedAndSharedJointsWithoutMagnitudeFiltering)
{
  auto b = std::make_shared<CertifiedBackend>(); b->frames = backend()->frames;
  // Joint 2 is common to all frames; joint 3 is unrelated. Duplicates are harmless.
  b->dependencies = {{"base", {}}, {"A", {0, 2, 2}}, {"B", {1, 2}},
    {"C", {4, 2}}, {"D", {5, 2}}};
  b->frames["C"].jacobian(2, 8) = 1; // Synthetic noise on a shared ancestor.
  b->frames["C"].jacobian(2, 9) = 1; // Synthetic noise on an unrelated joint.
  b->frames["C"].jacobian(2, 10) = 1e-14; // Valid, small coefficient must survive.
  SelfCollisionAvoidanceTask task; configure(task, parameters(), b);
  const size_t queries = b->dependency_calls;
  const auto out = task.update({}, *b); ASSERT_TRUE(out.active);
  EXPECT_EQ(b->dependency_calls, queries);
  EXPECT_DOUBLE_EQ(out.jacobian.leftCols(6).norm(), 0);
  EXPECT_DOUBLE_EQ(out.jacobian(0, 8), 0); EXPECT_DOUBLE_EQ(out.jacobian(0, 9), 0);
  EXPECT_NEAR(out.jacobian(0, 10), .5e-14, 1e-28);
  EXPECT_NE(task.build_status().status_message.find("active_joint_count=4"), std::string::npos);
  TaskComputation lower; lower.active = true;
  lower.jacobian = Eigen::MatrixXd::Zero(2, 14); lower.jacobian(0, 8) = 1; lower.jacobian(1, 9) = 1;
  lower.desired_velocity = Eigen::VectorXd::Constant(2, .2);
  HierarchySolver solver; solver.configure(14); solver.set_method(SolverMethod::kPinv);
  const auto free_joints = solver.solve({out, lower});
  EXPECT_NEAR(free_joints.generalized_velocity(8), .2, 1e-12);
  EXPECT_NEAR(free_joints.generalized_velocity(9), .2, 1e-12);
  // One uncertified frame requires a complete articulated fallback, not a partial mask.
  b->dependencies.erase("D"); configure(task, parameters(), b);
  const auto fallback = task.update({}, *b);
  EXPECT_NEAR(fallback.jacobian(0, 8), .5, 1e-12); EXPECT_NEAR(fallback.jacobian(0, 9), .5, 1e-12);
  // The original mock backend has no dependency API and preserves its distance derivative.
  auto unsupported = backend(); SelfCollisionAvoidanceTask original; configure(original, parameters(), unsupported);
  const auto full = original.update({}, *unsupported);
  EXPECT_NEAR(full.jacobian(0, 6), -.5, 1e-12); EXPECT_NEAR(full.jacobian(0, 11), .5, 1e-12);
  b->dependencies["A"] = {8}; EXPECT_THROW(configure(task, parameters(), b), std::invalid_argument);
}

TEST(SelfCollisionTask, ZeroJacobianPairStaysActiveAndReportsDeficit)
{
  auto b = std::make_shared<CertifiedBackend>(); b->frames = backend()->frames;
  b->dependencies = {{"base", {}}, {"A", {0}}, {"B", {0}}, {"C", {0}}, {"D", {0}}};
  SelfCollisionAvoidanceTask task; configure(task, parameters(), b);
  const auto out = task.update({}, *b); EXPECT_TRUE(out.active); EXPECT_FALSE(out.stop_arm_motion);
  EXPECT_DOUBLE_EQ(out.jacobian.norm(), 0); EXPECT_NEAR(out.desired_velocity(0), .08, 1e-12);
  WholeBodyCommand command; command.generalized_velocity = Eigen::VectorXd::Zero(14);
  task.observe_command(command, 1); const auto * snapshot = task.consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->metrics.active_pairs, 1u); EXPECT_EQ(snapshot->metrics.zero_jacobian_pairs, 1u);
  EXPECT_EQ(snapshot->metrics.active_joint_count, 0u);
  EXPECT_NEAR(snapshot->metrics.max_velocity_deficit, .08, 1e-12);
  EXPECT_TRUE(snapshot->capsules[0].highlighted);
  for (const auto & name : {"C", "D"}) {b->frames[name].pose.translation().z() = .205;}
  EXPECT_TRUE(task.update({}, *b).active);
  task.observe_command(command, 2); snapshot = task.consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->metrics.zero_jacobian_pairs, 1u);
  EXPECT_NEAR(snapshot->metrics.max_velocity_deficit, .005, 1e-12);
  for (const auto & name : {"C", "D"}) {b->frames[name].pose.translation().z() = .21;}
  EXPECT_FALSE(task.update({}, *b).active);
  task.observe_command(command, 3); snapshot = task.consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->metrics.zero_jacobian_pairs, 0u); EXPECT_DOUBLE_EQ(snapshot->metrics.max_velocity_deficit, 0);
}
TEST(SelfCollisionTask, InterpolatedDistanceJacobianMatchesFiniteDifferences)
{
  auto b=backend(); SelfCollisionAvoidanceTask task; configure(task,parameters(),b);
  const auto out=task.update({},*b);
  Eigen::VectorXd direction=Eigen::VectorXd::Zero(14);
  direction(6)=0.3; direction(7)=-0.2; direction(10)=0.1; direction(11)=0.4;
  constexpr double eps=1e-6;
  auto clearance=[&](double delta) {
      const auto perturbed=backend(delta*direction);
      const auto d=capsule_distance(perturbed->frames.at("A").pose.translation(),perturbed->frames.at("B").pose.translation(),
        perturbed->frames.at("C").pose.translation(),perturbed->frames.at("D").pose.translation(),0.04,0.04);
      EXPECT_GT(d.s,0); EXPECT_LT(d.s,1); EXPECT_GT(d.t,0); EXPECT_LT(d.t,1);
      return d.clearance;
    };
  EXPECT_NEAR((out.jacobian*direction)(0),(clearance(eps)-clearance(-eps))/(2*eps),1e-8);
}
TEST(SelfCollisionTask, MovingBaseDoesNotChangeInternalDistanceOrJacobian)
{
  Eigen::Isometry3d pose=Eigen::Isometry3d::Identity();
  pose.linear()=Eigen::AngleAxisd(0.7,V(1,2,3).normalized()).toRotationMatrix(); pose.translation()=V(2,3,4);
  auto a=backend(),b=backend(Eigen::VectorXd::Zero(14),pose);
  SelfCollisionAvoidanceTask ta,tb; configure(ta,parameters(),a); configure(tb,parameters(),b);
  auto oa=ta.update({},*a),ob=tb.update({},*b);
  EXPECT_LT((oa.jacobian-ob.jacobian).norm(),1e-12);
  EXPECT_LT((oa.desired_velocity-ob.desired_velocity).norm(),1e-12);
}
TEST(SelfCollisionTask, DegenerateThresholdAndAutomaticRecovery)
{
  auto b=backend(); auto p=parameters(); scalar(p,"degenerate_axis_distance",0.002);
  SelfCollisionAvoidanceTask task; configure(task,p,b);
  for(const auto & name : {"C","D"}) {b->frames[name].pose.translation().z()=0.001;}
  auto out=task.update({},*b);
  EXPECT_TRUE(out.stop_arm_motion); EXPECT_TRUE(out.jacobian.allFinite()); EXPECT_DOUBLE_EQ(out.jacobian.norm(),0);
  EXPECT_NE(task.build_status().status_message.find("degenerate_pairs=1"),std::string::npos);
  for(const auto & name : {"C","D"}) {b->frames[name].pose.translation().z()=0.003;}
  out=task.update({},*b); EXPECT_FALSE(out.stop_arm_motion); EXPECT_TRUE(out.active);
  task.set_enabled(false); out=task.update({},*b); EXPECT_FALSE(out.stop_arm_motion); EXPECT_FALSE(out.active);
}
TEST(SelfCollisionTask, InvalidRuntimeFramesStopArms)
{
  auto b=backend(); SelfCollisionAvoidanceTask task; configure(task,parameters(),b);
  b->frames.erase("A"); auto out=task.update({},*b);
  EXPECT_TRUE(out.stop_arm_motion); EXPECT_EQ(out.status_message,"invalid_collision_geometry");
}
TEST(SelfCollisionTask, InvalidRuntimeJacobiansStopArms)
{
  auto b=backend(); SelfCollisionAvoidanceTask task; configure(task,parameters(),b);
  b->frames["A"].jacobian(0,6)=std::numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(task.update({},*b).stop_arm_motion);
}
TEST(SelfCollisionTask, ExplicitPairsOnlyAndCanonicalExclusions)
{
  auto b=backend(); auto p=parameters();
  for (auto it = p.begin(); it != p.end();) {
    if (it->first.rfind("capsules.", 0) == 0) {it = p.erase(it);} else {++it;}
  }
  p["capsule_names"]=rclcpp::Parameter("capsule_names",std::vector<std::string>{"right_proximal","right_distal","left_proximal","left_distal"});
  capsule(p,"right_proximal","A","B"); capsule(p,"right_distal","B","A");
  capsule(p,"left_proximal","C","D"); capsule(p,"left_distal","D","C");
  p["ignored_collision_pairs"]=rclcpp::Parameter("ignored_collision_pairs",std::vector<std::string>{
    "right_proximal,right_distal","right_distal,right_proximal","left_proximal,left_distal"});
  SelfCollisionAvoidanceTask task; configure(task,p,b);
  EXPECT_EQ(task.capsules().size(),4u); EXPECT_EQ(task.ignored_pair_count(),2u); EXPECT_EQ(task.checked_pairs().size(),4u);
  p.erase("ignored_collision_pairs"); SelfCollisionAvoidanceTask all; configure(all,p,b);
  EXPECT_EQ(all.checked_pairs().size(),6u); EXPECT_TRUE(all.update({},*b).stop_arm_motion);
}
TEST(SelfCollisionTask, ReportsRequestedVersusAchievedSeparation)
{
  auto b = backend(); SelfCollisionAvoidanceTask task; configure(task, parameters(), b);
  task.update({}, *b);
  WholeBodyCommand command; command.generalized_velocity = Eigen::VectorXd::Zero(14);
  task.observe_command(command, 987);
  const auto * snapshot = task.consume_snapshot(); ASSERT_NE(snapshot, nullptr);
  EXPECT_EQ(snapshot->timestamp_ns, 987);
  EXPECT_NEAR(snapshot->metrics.max_velocity_deficit, 0.08, 1e-12);
  EXPECT_TRUE(snapshot->capsules[0].valid);
}
TEST(SelfCollisionTask, InvalidConfigurationIsRejected)
{
  auto b=backend();
  const std::vector<std::pair<std::string,double>> invalid={
    {"capsules.a.radius",0},{"capsules.a.radius",-1},{"capsules.a.radius",std::numeric_limits<double>::infinity()},
    {"safe_distance",-1},{"safe_distance",0.2},{"gain_scalar",-1},
    {"release_distance",.12},{"release_distance",.11},
    {"release_distance",std::numeric_limits<double>::infinity()},
    {"release_distance",std::numeric_limits<double>::quiet_NaN()},
    {"eps",0},{"eps",-.001},{"eps",.02},{"eps",std::numeric_limits<double>::quiet_NaN()},
    {"eps",std::numeric_limits<double>::infinity()},
    {"gain_scalar",std::numeric_limits<double>::quiet_NaN()},{"parallel_epsilon",0},
    {"parallel_epsilon",1},{"segment_length_epsilon",0},{"degenerate_axis_distance",0},
    {"max_repulsive_velocity",-1},{"capsule_publish_rate",0}};
  for(const auto & entry:invalid) {
    auto p=parameters(); scalar(p,entry.first,entry.second); SelfCollisionAvoidanceTask task;
    EXPECT_THROW(configure(task,p,b),std::invalid_argument)<<entry.first;
  }
  for(const auto & names:std::vector<std::vector<std::string>>{{},{"a","a"},{"invalid-name"}}) {
    auto p=parameters(); p["capsule_names"]=rclcpp::Parameter("capsule_names",names); SelfCollisionAvoidanceTask task;
    EXPECT_THROW(configure(task,p,b),std::invalid_argument);
  }
  for(const auto & pair: {"a,unknown","a,a","a,b,c","a"}) {
    auto p=parameters(); p["ignored_collision_pairs"]=rclcpp::Parameter("ignored_collision_pairs",std::vector<std::string>{pair});
    SelfCollisionAvoidanceTask task; EXPECT_THROW(configure(task,p,b),std::invalid_argument);
  }
  auto p=parameters(); p.erase("capsules.a.end_frame"); SelfCollisionAvoidanceTask task;
  EXPECT_THROW(configure(task,p,b),std::invalid_argument);
  p=parameters(); p["capsules.a.start_frame"]=rclcpp::Parameter("capsules.a.start_frame","unknown");
  EXPECT_THROW(configure(task,p,b),std::invalid_argument);
  // Equality at the admissible epsilon bound is rejected too.
  p=parameters(); scalar(p,"release_distance",.13); scalar(p,"eps",.13-.12);
  EXPECT_THROW(configure(task,p,b),std::invalid_argument);
}
TEST(SelfCollisionTask, RuntimeParameterPolicy)
{
  auto b=backend(); auto task=std::make_shared<SelfCollisionAvoidanceTask>(); configure(*task,parameters(),b);
  std::vector<std::shared_ptr<TaskBase>> tasks{task};
  EXPECT_FALSE(validate_collision_parameter_updates({rclcpp::Parameter("tasks.self_collision_avoidance.degenerate_axis_distance",0.1)},tasks).successful);
  EXPECT_TRUE(validate_collision_parameter_updates({rclcpp::Parameter("tasks.self_collision_avoidance.publish_capsules",false)},tasks).successful);
  EXPECT_FALSE(task->publish_capsules());
  EXPECT_FALSE(validate_collision_parameter_updates({rclcpp::Parameter("tasks.self_collision_avoidance.plane_names",std::vector<std::string>{"guard"})},tasks).successful);
  EXPECT_FALSE(validate_collision_parameter_updates({rclcpp::Parameter("tasks.self_collision_avoidance.planes.guard.reference_frame","base")},tasks).successful);
  EXPECT_FALSE(validate_collision_parameter_updates({rclcpp::Parameter("tasks.self_collision_avoidance.release_distance",.14)},tasks).successful);
  EXPECT_FALSE(validate_collision_parameter_updates({rclcpp::Parameter("tasks.self_collision_avoidance.eps",.0002)},tasks).successful);
  EXPECT_FALSE(validate_collision_parameter_updates({rclcpp::Parameter("tasks.self_collision_avoidance.gain_scalar",std::numeric_limits<double>::quiet_NaN())},tasks).successful);
}
TEST(SelfCollisionTask, KDLJointOrderingMatchesFiniteDifference)
{
  std::ostringstream urdf;
  urdf << "<robot name='test'><link name='base'/><link name='L0'/><link name='A'/>"
       << "<link name='L2'/><link name='B'/><link name='R0'/><link name='C'/>"
       << "<link name='R2'/><link name='D'/><link name='unused_mesh'>"
       << "<collision><geometry><mesh filename='package://missing_test_package/nonexistent.dae'/>"
       << "</geometry></collision></link><joint name='mesh_fixed' type='fixed'><parent link='base'/>"
       << "<child link='unused_mesh'/></joint>";
  auto joint = [&](const char * name,const char * parent,const char * child,const char * xyz,const char * axis) {
      urdf << "<joint name='" << name << "' type='revolute'><parent link='" << parent
           << "'/><child link='" << child << "'/><origin xyz='" << xyz << "'/><axis xyz='" << axis
           << "'/><limit lower='-3' upper='3' effort='1' velocity='1'/></joint>";
    };
  joint("l0","base","L0","-0.2 0 0","0 0 1");
  joint("l1","L0","A","0.1 0 0","0 1 0");
  joint("l2","A","L2","0.1 0 0","0 0 1");
  joint("l3","L2","B","0.1 0 0","0 1 0");
  joint("r0","base","R0","0 -0.2 0.15","1 0 0");
  joint("r1","R0","C","0 0.1 0","0 0 1");
  joint("r2","C","R2","0 0.1 0","1 0 0");
  joint("r3","R2","D","0 0.1 0","0 0 1");
  urdf << "</robot>";
  auto m=model(); auto kdl=std::make_shared<KDLKinematicsBackend>();
  kdl->configure(*m,urdf.str(),rclcpp::get_logger("collision_kdl_test"));
  EXPECT_TRUE(kdl->collision_capsules().empty());
  WholeBodyState state;
  state.joint_positions.resize(8); state.joint_positions << .15,-.12,.25,-.06,.08,-.16,.22,-.09;
  state.joint_velocities=Eigen::VectorXd::Zero(8);
  kdl->update(state);
  std::vector<size_t> dependencies{99};
  ASSERT_TRUE(kdl->frame_joint_dependencies("base", dependencies)); EXPECT_TRUE(dependencies.empty());
  ASSERT_TRUE(kdl->frame_joint_dependencies("A", dependencies));
  EXPECT_EQ(dependencies, (std::vector<size_t>{0, 1}));
  ASSERT_TRUE(kdl->frame_joint_dependencies("B", dependencies));
  EXPECT_EQ(dependencies, (std::vector<size_t>{0, 1, 2, 3}));
  ASSERT_TRUE(kdl->frame_joint_dependencies("C", dependencies));
  EXPECT_EQ(dependencies, (std::vector<size_t>{4, 5}));
  ASSERT_TRUE(kdl->clone()->frame_joint_dependencies("D", dependencies));
  EXPECT_EQ(dependencies, (std::vector<size_t>{4, 5, 6, 7}));
  ASSERT_TRUE(kdl->frame_joint_dependencies("unused_mesh", dependencies)); EXPECT_TRUE(dependencies.empty());
  EXPECT_FALSE(kdl->frame_joint_dependencies("missing", dependencies)); EXPECT_TRUE(dependencies.empty());
  auto p=parameters(); scalar(p,"activation_distance",10.0);
  SelfCollisionAvoidanceTask task;
  task.configure("self_collision_avoidance","task_priority_kinematic_control/SelfCollisionAvoidanceTask",p,TaskContext{m,kdl});
  const auto output=task.update(state,*kdl); ASSERT_TRUE(output.active); ASSERT_FALSE(output.stop_arm_motion);
  auto clearance = [&](const WholeBodyState & s) {
      const V a=kdl->compute_frame_state(s,"A").pose.translation();
      const V b=kdl->compute_frame_state(s,"B").pose.translation();
      const V c=kdl->compute_frame_state(s,"C").pose.translation();
      const V d=kdl->compute_frame_state(s,"D").pose.translation();
      return capsule_distance(a,b,c,d,0.04,0.04).clearance;
    };
  for(Eigen::Index i=0;i<8;++i) {
    auto plus=state,minus=state;
    plus.joint_positions(i)+=1e-6; minus.joint_positions(i)-=1e-6;
    EXPECT_NEAR(output.jacobian(0,6+i),(clearance(plus)-clearance(minus))/2e-6,1e-7) << "joint " << i;
  }
  // Same-arm capsules: ancestors l0/l1 are a shared rigid motion, l2 changes distance.
  capsule(p, "a", "A", "A"); capsule(p, "b", "B", "B");
  task.configure("self_collision_avoidance", "task_priority_kinematic_control/SelfCollisionAvoidanceTask", p, TaskContext{m, kdl});
  auto masked = task.update(state, *kdl); ASSERT_TRUE(masked.active); ASSERT_FALSE(masked.stop_arm_motion);
  EXPECT_DOUBLE_EQ(masked.jacobian.leftCols(8).norm(), 0); EXPECT_DOUBLE_EQ(masked.jacobian.rightCols(4).norm(), 0);
  EXPECT_GT(std::abs(masked.jacobian(0, 8)), 1e-3);
  for (Eigen::Index i = 0; i < 8; ++i) {
    auto plus = state, minus = state; plus.joint_positions(i) += 1e-6; minus.joint_positions(i) -= 1e-6;
    const auto separation = [&](const WholeBodyState & s) {
        return (kdl->compute_frame_state(s, "B").pose.translation() -
               kdl->compute_frame_state(s, "A").pose.translation()).norm() - .08;
      };
    EXPECT_NEAR(masked.jacobian(0, 6 + i), (separation(plus) - separation(minus)) / 2e-6, 1e-8);
  }
  // A base-referenced plane uses one arm only; an A-referenced plane excludes l0/l1.
  auto plane = plane_parameters(); scalar(plane, "activation_distance", 10);
  plane["planes.guard.normal_axis"] = rclcpp::Parameter("planes.guard.normal_axis", "y");
  plane["planes.guard.allowed_side"] = rclcpp::Parameter("planes.guard.allowed_side", "negative");
  scalar(plane, "planes.guard.position", .5);
  for (const auto & reference : {"base", "A"}) {
    SCOPED_TRACE(reference);
    plane["planes.guard.reference_frame"] = rclcpp::Parameter("planes.guard.reference_frame", reference);
    task.configure("self_collision_avoidance", "task_priority_kinematic_control/SelfCollisionAvoidanceTask", plane, TaskContext{m, kdl});
    masked = task.update(state, *kdl); ASSERT_TRUE(masked.active); ASSERT_FALSE(masked.stop_arm_motion);
    EXPECT_DOUBLE_EQ(masked.jacobian.rightCols(4).norm(), 0);
    if (std::string(reference) == "A") {
      EXPECT_DOUBLE_EQ(masked.jacobian.leftCols(8).norm(), 0);
      EXPECT_GT(std::abs(masked.jacobian(0, 8)), 1e-3);
    }
    const auto plane_clearance = [&](const WholeBodyState & s) {
        const auto inverse = kdl->compute_frame_state(s, reference).pose.inverse();
        return capsule_plane_distance(inverse * kdl->compute_frame_state(s, "A").pose.translation(),
          inverse * kdl->compute_frame_state(s, "B").pose.translation(), .04, 1, -1, .5,
          {-.5, -.5}, {.5, .5}).clearance;
      };
    for (Eigen::Index i = 0; i < 8; ++i) {
      auto plus = state, minus = state; plus.joint_positions(i) += 1e-6; minus.joint_positions(i) -= 1e-6;
      EXPECT_NEAR(masked.jacobian(0, 6 + i), (plane_clearance(plus) - plane_clearance(minus)) / 2e-6, 1e-8);
    }
  }
}

class ManagerTest : public ::testing::Test
{
  void SetUp() override {if(!rclcpp::ok()) {rclcpp::init(0,nullptr);}}
};
TEST_F(ManagerTest, DeclaresPlaneOverridesAndDefaultsWithoutRobotSpecificFrames)
{
  const std::string prefix = "tasks.self_collision_avoidance.";
  std::vector<rclcpp::Parameter> overrides;
  for (const auto & parameter : plane_parameters()) {
    overrides.emplace_back(prefix + parameter.first, parameter.second.get_parameter_value());
  }
  rclcpp::NodeOptions options; options.parameter_overrides(overrides);
  auto node = std::make_shared<rclcpp::Node>("plane_declaration_test", options);
  declare_collision_parameters(node->get_node_parameters_interface(), prefix);
  EXPECT_EQ(node->get_parameter(prefix + "plane_names").as_string_array(), std::vector<std::string>{"guard"});
  EXPECT_EQ(node->get_parameter(prefix + "planes.guard.reference_frame").as_string(), "");
  EXPECT_EQ(node->get_parameter(prefix + "planes.guard.normal_axis").as_string(), "z");
  EXPECT_EQ(node->get_parameter(prefix + "planes.guard.bounds_min").as_double_array(), (std::vector<double>{-.5, -.5}));
  EXPECT_EQ(node->get_parameter(prefix + "planes.guard.checked_capsules").as_string_array(), std::vector<std::string>{"a"});
  // An absent list must also be declared as a typed empty array.
  auto empty = std::make_shared<rclcpp::Node>("plane_empty_declaration_test");
  declare_collision_parameters(empty->get_node_parameters_interface(), prefix);
  EXPECT_TRUE(empty->get_parameter(prefix + "plane_names").as_string_array().empty());
  EXPECT_DOUBLE_EQ(empty->get_parameter(prefix + "release_distance").as_double(), .12 + .01);
  EXPECT_DOUBLE_EQ(empty->get_parameter(prefix + "eps").as_double(), .0001);
}
TEST_F(ManagerTest, ReleaseDefaultUsesEffectiveActivationOverride)
{
  const std::string prefix = "tasks.self_collision_avoidance.";
  rclcpp::NodeOptions options;
  options.parameter_overrides({rclcpp::Parameter(prefix + "activation_distance", .05)});
  auto node = std::make_shared<rclcpp::Node>("hysteresis_declaration_default", options);
  declare_collision_parameters(node->get_node_parameters_interface(), prefix);
  EXPECT_DOUBLE_EQ(node->get_parameter(prefix + "release_distance").as_double(), .05 + .01);
  options.parameter_overrides({rclcpp::Parameter(prefix + "activation_distance", .05),
    rclcpp::Parameter(prefix + "release_distance", .08), rclcpp::Parameter(prefix + "eps", .0002)});
  auto explicit_node = std::make_shared<rclcpp::Node>("hysteresis_declaration_override", options);
  declare_collision_parameters(explicit_node->get_node_parameters_interface(), prefix);
  EXPECT_DOUBLE_EQ(explicit_node->get_parameter(prefix + "release_distance").as_double(), .08);
  EXPECT_DOUBLE_EQ(explicit_node->get_parameter(prefix + "eps").as_double(), .0002);
}
TEST_F(ManagerTest, FinalCommandsStopBothArmsAndPreserveBase)
{
  auto b=backend(); for(const auto & name : {"C","D"}) {b->frames[name].pose.translation().z()=0;}
  auto node=std::make_shared<rclcpp::Node>("collision_manager_test");
  node->declare_parameter("task_ids",std::vector<std::string>{"self_collision_avoidance"});
  for(const auto & parameter:parameters()) {node->declare_parameter("tasks.self_collision_avoidance."+parameter.first,parameter.second.get_parameter_value());}
  TaskManager manager(node->get_node_parameters_interface(),node->get_logger());
  manager.configure(TaskContext{model(),b}); manager.update_all_into({},*b);
  WholeBodyCommand command; command.generalized_velocity=Eigen::VectorXd::Ones(14);
  manager.finalize_command(command,123);
  EXPECT_DOUBLE_EQ(command.generalized_velocity.head(6).sum(),6);
  EXPECT_DOUBLE_EQ(command.generalized_velocity.tail(8).norm(),0);
  for(const auto & name : {"C","D"}) {b->frames[name].pose.translation().z()=0.1;}
  manager.update_all_into({},*b); command.generalized_velocity.setOnes(); manager.finalize_command(command,124);
  EXPECT_DOUBLE_EQ(command.generalized_velocity.tail(8).sum(),8);
}
TEST_F(ManagerTest, ObserverPublishesAndClearsMarkersOnItsOwnExecutor)
{
  auto b = backend();
  auto task = std::make_shared<SelfCollisionAvoidanceTask>(); configure(*task, plane_parameters(), b);
  auto node = std::make_shared<rclcpp::Node>("capsule_observer_test");
  bool received = false, cleared = false;
  auto subscription = node->create_subscription<visualization_msgs::msg::MarkerArray>(
    std::string(node->get_fully_qualified_name()) + "/self_collision_avoidance/capsules",
    rclcpp::QoS(10), // Reliable, matching the default RViz display QoS.
    [&](visualization_msgs::msg::MarkerArray::SharedPtr message) {
      if (message->markers.size() != 7) {return;}
      const bool deleted = std::all_of(message->markers.begin(), message->markers.end(),
        [](const auto & marker) {return marker.action == visualization_msgs::msg::Marker::DELETE;});
      if (deleted) {cleared = true;} else {
        received = true;
        EXPECT_EQ(message->markers[0].header.frame_id, "base");
        EXPECT_NEAR(message->markers[0].scale.x, 0.08, 1e-12);
        EXPECT_EQ(message->markers.back().ns, "self_collision_avoidance/planes/guard");
        EXPECT_EQ(message->markers.back().type, visualization_msgs::msg::Marker::TRIANGLE_LIST);
      }
    });
  TaskComputation output; task->prepare_computation(output);
  WholeBodyCommand command; command.generalized_velocity = Eigen::VectorXd::Zero(14);
  // Publish snapshots before the observer/subscriber runs; these never wait for DDS.
  for (int i = 0; i < 1000; ++i) {
    task->update_into({}, *b, output); task->observe_command(command, i);
  }
  CapsuleObserver observer({task}, node->get_fully_qualified_name(), node->get_node_base_interface()->get_context());
  auto wait_for = [&](const bool & condition) {
      const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(4);
      while (!condition && std::chrono::steady_clock::now() < deadline) {
        rclcpp::spin_some(node); std::this_thread::sleep_for(std::chrono::milliseconds(10));
      }
    };
  wait_for(received); ASSERT_TRUE(received);
  task->set_publish_capsules(false);
  wait_for(cleared); EXPECT_TRUE(cleared);
  received = false; cleared = false;
  task->set_publish_capsules(true);
  wait_for(received); ASSERT_TRUE(received);
  task->set_enabled(false); task->update_into({}, *b, output); task->observe_command(command, 1001);
  wait_for(cleared); EXPECT_TRUE(cleared);
}
}  // namespace
}  // namespace task_priority_kinematic_control
