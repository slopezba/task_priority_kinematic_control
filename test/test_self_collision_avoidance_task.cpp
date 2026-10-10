#include "task_priority_kinematic_control/tasks/self_collision_avoidance_task.hpp"
#include "task_priority_kinematic_control/core/task_manager.hpp"
#include "task_priority_kinematic_control/geometry/capsule_geometry.hpp"
#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"
#include "task_priority_kinematic_control/core/capsule_observer.hpp"
#include <gtest/gtest.h>
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
  EXPECT_TRUE(out.active); EXPECT_NEAR(out.desired_velocity(0),0.02,1e-12); EXPECT_DOUBLE_EQ(out.error(0),0);
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
}
TEST(SelfCollisionTask, RuntimeParameterPolicy)
{
  auto b=backend(); auto task=std::make_shared<SelfCollisionAvoidanceTask>(); configure(*task,parameters(),b);
  std::vector<std::shared_ptr<TaskBase>> tasks{task};
  EXPECT_FALSE(validate_collision_parameter_updates({rclcpp::Parameter("tasks.self_collision_avoidance.degenerate_axis_distance",0.1)},tasks).successful);
  EXPECT_TRUE(validate_collision_parameter_updates({rclcpp::Parameter("tasks.self_collision_avoidance.publish_capsules",false)},tasks).successful);
  EXPECT_FALSE(task->publish_capsules());
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
}

class ManagerTest : public ::testing::Test
{
  void SetUp() override {if(!rclcpp::ok()) {rclcpp::init(0,nullptr);}}
};
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
  auto task = std::make_shared<SelfCollisionAvoidanceTask>(); configure(*task, parameters(), b);
  auto node = std::make_shared<rclcpp::Node>("capsule_observer_test");
  bool received = false, cleared = false;
  auto subscription = node->create_subscription<visualization_msgs::msg::MarkerArray>(
    std::string(node->get_fully_qualified_name()) + "/self_collision_avoidance/capsules",
    rclcpp::QoS(10), // Reliable, matching the default RViz display QoS.
    [&](visualization_msgs::msg::MarkerArray::SharedPtr message) {
      if (message->markers.size() != 6) {return;}
      const bool deleted = std::all_of(message->markers.begin(), message->markers.end(),
        [](const auto & marker) {return marker.action == visualization_msgs::msg::Marker::DELETE;});
      if (deleted) {cleared = true;} else {
        received = true;
        EXPECT_EQ(message->markers[0].header.frame_id, "base");
        EXPECT_NEAR(message->markers[0].scale.x, 0.08, 1e-12);
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
}
}  // namespace
}  // namespace task_priority_kinematic_control
