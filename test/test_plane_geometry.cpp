#include "task_priority_kinematic_control/geometry/plane_geometry.hpp"
#include <gtest/gtest.h>
#include <cmath>
#include <limits>

namespace task_priority_kinematic_control
{
namespace
{
using V = Eigen::Vector3d;
const Eigen::Vector2d minimum(-0.2, -0.3), maximum(0.2, 0.3);

TEST(PlaneGeometry, AxesSidesAndSignedClearance)
{
  EXPECT_EQ(plane_tangent_axes(0), (std::array<int, 2>{1, 2}));
  EXPECT_EQ(plane_tangent_axes(1), (std::array<int, 2>{0, 2}));
  EXPECT_EQ(plane_tangent_axes(2), (std::array<int, 2>{0, 1}));
  for (int axis = 0; axis < 3; ++axis) {
    for (const double sign : {1.0, -1.0}) {
      V a = V::Zero(), b = V::Zero();
      a(axis) = sign * 0.10; b(axis) = sign * 0.20;
      const auto d = capsule_plane_distance(a, b, 0.025, axis, sign, 0, minimum, maximum);
      ASSERT_TRUE(d.valid); ASSERT_TRUE(d.intersects);
      EXPECT_NEAR(d.clearance, 0.075, 1e-12); EXPECT_DOUBLE_EQ(d.s, 0);
      EXPECT_NEAR(d.gradient_a(axis), sign, 1e-12); EXPECT_DOUBLE_EQ(d.gradient_b.norm(), 0);
      a(axis) = -sign * 0.1;
      const auto crossed = capsule_plane_distance(a, b, 0.025, axis, sign, 0, minimum, maximum);
      EXPECT_TRUE(crossed.valid); EXPECT_NEAR(crossed.clearance, -0.125, 1e-12);
      EXPECT_NEAR(crossed.gradient_a(axis), sign, 1e-12);
    }
  }
}

TEST(PlaneGeometry, FiniteBoundsIncludeRadiusAndHandlePointsAndParallelSegments)
{
  auto d = capsule_plane_distance({0, 0.23, 0}, {0, 0.23, 0.1}, 0.025, 0, 1, 0, minimum, maximum);
  EXPECT_TRUE(d.valid); EXPECT_FALSE(d.intersects);
  d = capsule_plane_distance({0, 0.225, 0}, {0, 0.225, 0}, 0.025, 0, 1, 0, minimum, maximum);
  EXPECT_TRUE(d.valid); EXPECT_TRUE(d.intersects); EXPECT_DOUBLE_EQ(d.clearance, -0.025);
  d = capsule_plane_distance({0.1, -0.5, 0}, {0.2, 0.5, 0}, 0.025, 0, 1, 0, minimum, maximum);
  EXPECT_TRUE(d.valid); EXPECT_TRUE(d.intersects); EXPECT_NEAR(d.s, 0.275, 1e-12);
  EXPECT_NEAR(d.clearance, 0.1025, 1e-12);
  d = capsule_plane_distance({0.1, 0, 0.4}, {0.2, 0.1, 0.4}, 0.025, 0, 1, 0, minimum, maximum);
  EXPECT_TRUE(d.valid); EXPECT_FALSE(d.intersects);
}

TEST(PlaneGeometry, ClippingGradientsMatchFiniteDifferencesForBothEnds)
{
  for (const double sign : {1.0, -1.0}) {
    const V a(0.10, -0.5, -0.1), b(0.20, 0.5, 0.1);
    const auto d = capsule_plane_distance(a, b, 0.025, 0, sign, 0, minimum, maximum);
    ASSERT_TRUE(d.valid); ASSERT_TRUE(d.intersects); EXPECT_GT(d.s, 0); EXPECT_LT(d.s, 1);
    // Tangential derivatives are nonzero because the minimum is on a clipped edge.
    EXPECT_GT(std::abs(d.gradient_a.y()) + std::abs(d.gradient_b.y()), 0);
    for (int endpoint = 0; endpoint < 2; ++endpoint) {
      for (int axis = 0; axis < 3; ++axis) {
        V ap = a, am = a, bp = b, bm = b;
        (endpoint == 0 ? ap : bp)(axis) += 1e-6;
        (endpoint == 0 ? am : bm)(axis) -= 1e-6;
        const auto plus = capsule_plane_distance(ap, bp, 0.025, 0, sign, 0, minimum, maximum);
        const auto minus = capsule_plane_distance(am, bm, 0.025, 0, sign, 0, minimum, maximum);
        ASSERT_TRUE(plus.intersects); ASSERT_TRUE(minus.intersects);
        EXPECT_NEAR((plus.clearance - minus.clearance) / 2e-6,
          (endpoint == 0 ? d.gradient_a : d.gradient_b)(axis), 1e-9);
      }
    }
  }
}

TEST(PlaneGeometry, RejectsInvalidAndUnrepresentableInputs)
{
  const V a(0.1, 0, 0), b(0.2, 0, 0);
  for (double radius : {0.0, -1.0, std::numeric_limits<double>::infinity()}) {
    EXPECT_FALSE(capsule_plane_distance(a, b, radius, 0, 1, 0, minimum, maximum).valid);
  }
  EXPECT_FALSE(capsule_plane_distance(a, b, .025, 3, 1, 0, minimum, maximum).valid);
  EXPECT_FALSE(capsule_plane_distance(a, b, .025, 0, 0, 0, minimum, maximum).valid);
  EXPECT_FALSE(capsule_plane_distance(a, b, .025, 0, 1, 0, maximum, minimum).valid);
  V bad = a; bad.x() = std::numeric_limits<double>::quiet_NaN();
  EXPECT_FALSE(capsule_plane_distance(bad, b, .025, 0, 1, 0, minimum, maximum).valid);
  const double limit = std::numeric_limits<double>::max();
  EXPECT_FALSE(capsule_plane_distance({-limit, 0, 0}, {-limit, 0, 0},
    .025, 0, 1, limit, minimum, maximum).valid);
}
}  // namespace
}  // namespace task_priority_kinematic_control
