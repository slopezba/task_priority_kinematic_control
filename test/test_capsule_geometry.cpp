#include "task_priority_kinematic_control/geometry/capsule_geometry.hpp"
#include <gtest/gtest.h>
#include <Eigen/Geometry>
#include <limits>

namespace task_priority_kinematic_control
{
namespace
{
using V = Eigen::Vector3d;
struct Case {const char * name; V a, b, c, d; double distance, ra = 0, rb = 0;};
const Case cases[] = {
  {"PerpendicularInterior", {-1,0,0},{1,0,0},{0,-1,1},{0,1,1},1},
  {"ParallelOverlappingProjections", {0,0,0},{2,0,0},{1,1,0},{3,1,0},1},
  {"ParallelDisjointProjections", {0,0,0},{1,0,0},{2,1,0},{3,1,0},std::sqrt(2.)},
  {"NearlyParallel", {0,0,0},{1,0,0},{0,0.2,0},{1,0.200001,0},0.2},
  {"OneEndpoint", {0,0,0},{1,0,0},{2,-1,0},{2,1,0},1},
  {"TwoEndpoints", {0,0,0},{1,0,0},{2,1,0},{3,2,0},std::sqrt(2.)},
  {"Intersection", {-1,0,0},{1,0,0},{0,-1,0},{0,1,0},0},
  {"OnePoint", {0,0,0},{0,0,0},{1,-1,0},{1,1,0},1},
  {"TwoPoints", {0,0,0},{0,0,0},{1,2,2},{1,2,2},3},
  {"PositiveClearance", {0,0,0},{1,0,0},{0,0.2,0},{1,0.2,0},0.2,0.04,0.04},
  {"TouchingCapsules", {0,0,0},{1,0,0},{0,0.08,0},{1,0.08,0},0.08,0.04,0.04},
  {"OverlappingCapsules", {0,0,0},{1,0,0},{0,0.06,0},{1,0.06,0},0.06,0.04,0.04},
  {"IdenticalSegments", {0,0,0},{1,0,0},{0,0,0},{1,0,0},0},
  {"ReversedEndpoints", {1,0,0},{0,0,0},{1,1,0},{0,1,0},1},
  {"ShortNonzeroSegments", {-1e-8,0,0},{1e-8,0,0},{0,-1e-8,1e-8},{0,1e-8,1e-8},1e-8},
  {"LargeOffsets", {1e9,1e9,1e9},{1e9+1,1e9,1e9},{1e9,1e9+0.001,1e9},{1e9+1,1e9+0.001,1e9},(1e9+0.001)-1e9}
};
class CapsuleGeometry : public ::testing::TestWithParam<Case> {};
TEST_P(CapsuleGeometry, FiniteConstrainedMinimum)
{
  const auto & c = GetParam();
  const auto out = capsule_distance(c.a,c.b,c.c,c.d,c.ra,c.rb);
  ASSERT_TRUE(out.valid);
  EXPECT_TRUE(out.p.allFinite()); EXPECT_TRUE(out.q.allFinite());
  EXPECT_TRUE(std::isfinite(out.axis_distance)); EXPECT_TRUE(std::isfinite(out.clearance));
  EXPECT_GE(out.s,0); EXPECT_LE(out.s,1); EXPECT_GE(out.t,0); EXPECT_LE(out.t,1);
  EXPECT_NEAR(out.axis_distance,c.distance,1e-12);
  EXPECT_NEAR(out.clearance,c.distance-c.ra-c.rb,1e-12);
  EXPECT_LT((out.p-((1-out.s)*c.a+out.s*c.b)).norm(),1e-6);
  EXPECT_LT((out.q-((1-out.t)*c.c+out.t*c.d)).norm(),1e-6);
  const auto symmetric = capsule_distance(c.c,c.d,c.a,c.b,c.rb,c.ra);
  EXPECT_NEAR(out.axis_distance,symmetric.axis_distance,1e-12);
  const auto reversed = capsule_distance(c.b,c.a,c.d,c.c,c.ra,c.rb);
  EXPECT_NEAR(out.axis_distance,reversed.axis_distance,1e-12);
}
INSTANTIATE_TEST_SUITE_P(RequiredCases,CapsuleGeometry,::testing::ValuesIn(cases),
  [](const ::testing::TestParamInfo<Case> & info) {return info.param.name;});
TEST(CapsuleGeometryExtra, InteriorParameters)
{
  const auto out = capsule_distance({-1,0,0},{1,0,0},{0,-1,1},{0,1,1},0,0);
  EXPECT_DOUBLE_EQ(out.s,0.5); EXPECT_DOUBLE_EQ(out.t,0.5);
}
TEST(CapsuleGeometryExtra, ScaleIndependentParallelDetection)
{
  for (double length : {1e-5,1.,1e5}) {
    const auto out = capsule_distance({0,0,0},{length,0,0}, {0,0.2,0},{length,0.2+length*1e-6,0},0,0);
    ASSERT_TRUE(out.valid); EXPECT_NEAR(out.axis_distance,0.2,1e-12);
  }
}
TEST(CapsuleGeometryExtra, InvalidInputIsReported)
{
  const double nan = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(capsule_distance({nan,0,0},{0,0,0},{0,0,0},{1,0,0},0,0).valid);
  EXPECT_FALSE(capsule_distance(V::Zero(),V::Zero(),V::Zero(),V::Zero(),-1,0).valid);
}
TEST(CapsuleGeometryExtra, ExtremeFiniteInputsRemainFinite)
{
  const auto out = capsule_distance({-1e308,0,0},{-1e308,0,0},{1e308,0,0},{1e308,0,0},0.04,0.04);
  EXPECT_TRUE(out.valid); EXPECT_TRUE(std::isfinite(out.axis_distance)); EXPECT_TRUE(std::isfinite(out.clearance));
}
}  // namespace
}  // namespace task_priority_kinematic_control
