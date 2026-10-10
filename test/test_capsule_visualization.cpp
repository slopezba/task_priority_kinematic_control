#include "task_priority_kinematic_control/core/capsule_observer.hpp"
#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <atomic>
#include <thread>

namespace task_priority_kinematic_control
{
namespace
{
using Marker=visualization_msgs::msg::Marker;
TEST(CapsuleVisualization, PositionsScalesOrientationAndStableIdentity)
{
  ManualCapsule capsule; capsule.name="right"; capsule.start_frame="cirtesub/alpha_right/start"; capsule.radius=0.04;
  CollisionSnapshot snapshot; snapshot.timestamp_ns=1230000000;
  CapsulePose pose; pose.a={1,2,3}; pose.b={3,2,3}; pose.valid=true; snapshot.capsules.push_back(pose);
  const auto out=capsule_markers({capsule},snapshot,"base",1e-9);
  ASSERT_EQ(out.markers.size(),3u);
  const auto & cylinder=out.markers[0];
  EXPECT_EQ(cylinder.type,Marker::CYLINDER); EXPECT_EQ(cylinder.header.frame_id,"base");
  EXPECT_EQ(cylinder.header.stamp.sec,1); EXPECT_EQ(cylinder.header.stamp.nanosec,230000000u);
  EXPECT_DOUBLE_EQ(cylinder.pose.position.x,2); EXPECT_DOUBLE_EQ(cylinder.pose.position.y,2);
  EXPECT_DOUBLE_EQ(cylinder.scale.x,0.08); EXPECT_DOUBLE_EQ(cylinder.scale.y,0.08); EXPECT_DOUBLE_EQ(cylinder.scale.z,2);
  const auto & q=cylinder.pose.orientation;
  const Eigen::Quaterniond rotation(q.w,q.x,q.y,q.z);
  EXPECT_LT((rotation*Eigen::Vector3d::UnitZ()-Eigen::Vector3d::UnitX()).norm(),1e-12);
  EXPECT_FLOAT_EQ(cylinder.color.b,1.0f);
  EXPECT_DOUBLE_EQ(out.markers[1].pose.position.x,1); EXPECT_DOUBLE_EQ(out.markers[2].pose.position.x,3);
  for(size_t i=0;i<3;++i) {EXPECT_EQ(out.markers[i].id,static_cast<int>(i)); EXPECT_EQ(out.markers[i].ns,"self_collision_avoidance/right");}
  snapshot.capsules[0].highlighted=true;
  const auto active=capsule_markers({capsule},snapshot,"base",1e-9);
  EXPECT_FLOAT_EQ(active.markers[0].color.r,1); EXPECT_FLOAT_EQ(active.markers[0].color.b,0);
}
TEST(CapsuleVisualization, ZeroLengthAndAntiparallelOrientation)
{
  ManualCapsule capsule; capsule.name="left"; capsule.start_frame="alpha_left/A"; capsule.radius=0.05;
  CollisionSnapshot snapshot; snapshot.capsules.resize(1); snapshot.capsules[0].valid=true;
  auto out=capsule_markers({capsule},snapshot,"base",1e-9);
  EXPECT_EQ(out.markers[0].action,Marker::DELETE); EXPECT_EQ(out.markers[1].action,Marker::ADD); EXPECT_EQ(out.markers[2].action,Marker::DELETE);
  EXPECT_DOUBLE_EQ(out.markers[1].scale.z,0.1); EXPECT_FLOAT_EQ(out.markers[1].color.g,1);
  snapshot.capsules[0].b={0,0,-1}; out=capsule_markers({capsule},snapshot,"base",1e-9);
  const auto & q=out.markers[0].pose.orientation;
  EXPECT_LT((Eigen::Quaterniond(q.w,q.x,q.y,q.z)*Eigen::Vector3d::UnitZ()+Eigen::Vector3d::UnitZ()).norm(),1e-12);
}
TEST(CapsuleVisualization, ClearOnlyOwnedMarkers)
{
  ManualCapsule capsule; capsule.name="one"; capsule.radius=0.04;
  const auto out=delete_capsule_markers({capsule},"base");
  ASSERT_EQ(out.markers.size(),3u);
  for(const auto & marker:out.markers) {EXPECT_EQ(marker.action,Marker::DELETE); EXPECT_EQ(marker.ns,"self_collision_avoidance/one");}
}
struct Snapshot {std::vector<uint64_t> values;};
TEST(CollisionSnapshots, SlowConsumerNeverHoldsProducerSlot)
{
  SnapshotBuffer<Snapshot> buffer;
  buffer.initialize([](Snapshot & s) {s.values.resize(64);});
  const auto * first=buffer.consume(); EXPECT_EQ(first,nullptr);
  auto & initial=buffer.writable(); std::fill(initial.values.begin(),initial.values.end(),1); buffer.publish();
  const auto * held=buffer.consume(); ASSERT_NE(held,nullptr);
  std::thread producer([&] {
      for(uint64_t i=2;i<100002;++i) {
        auto & writable=buffer.writable();
        std::fill(writable.values.begin(),writable.values.end(),i); buffer.publish();
      }
    });
  producer.join(); // Completes even though the consumer retains its first snapshot throughout.
  for(auto v:held->values) {EXPECT_EQ(v,1u);}
  const auto * latest=buffer.consume(); ASSERT_NE(latest,nullptr);
  for(auto v:latest->values) {EXPECT_EQ(v,100001u);}
}
TEST(CollisionSnapshots, ConcurrentSnapshotsAreCoherent)
{
  SnapshotBuffer<Snapshot> buffer; buffer.initialize([](Snapshot & s) {s.values.resize(128);});
  std::atomic_bool done{false};
  std::thread producer([&] {
      for(uint64_t i=1;i<=50000;++i) {
        auto & writable=buffer.writable(); std::fill(writable.values.begin(),writable.values.end(),i); buffer.publish();
      }
      done.store(true);
    });
  uint64_t previous=0;
  do {
    if(const auto * s=buffer.consume()) {
      const auto value=s->values.front(); EXPECT_GE(value,previous); previous=value;
      for(auto element:s->values) {EXPECT_EQ(element,value);}
    }
  } while(!done.load());
  producer.join();
}
}  // namespace
}  // namespace task_priority_kinematic_control
