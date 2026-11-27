#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"

#include <gtest/gtest.h>

using task_priority_kinematic_control::HierarchySolver;
using task_priority_kinematic_control::SolverMethod;
using task_priority_kinematic_control::TaskComputation;

TEST(HierarchySolver, SolvesSingleTask)
{
  HierarchySolver solver;
  solver.configure(2);
  solver.set_method(SolverMethod::kPinv);

  TaskComputation task;
  task.active = true;
  task.jacobian = Eigen::MatrixXd::Identity(2, 2);
  task.desired_velocity = Eigen::Vector2d(1.0, -1.0);
  task.error = task.desired_velocity;

  const auto command = solver.solve({task});
  ASSERT_EQ(command.generalized_velocity.size(), 2);
  EXPECT_NEAR(command.generalized_velocity(0), 1.0, 1e-3);
  EXPECT_NEAR(command.generalized_velocity(1), -1.0, 1e-3);
}

TEST(HierarchySolver, ScaledPrefixPreservesPriorityThroughRankChanges)
{
  HierarchySolver solver;
  solver.configure(3);
  solver.set_method(SolverMethod::kPinv);
  TaskComputation high, nominal;
  high.active = true;
  high.jacobian.resize(1,3);
  high.jacobian << 1.0, 1.0, 0.0;
  high.desired_velocity = Eigen::VectorXd::Constant(1,3.0);
  HierarchySolver::Prefix prefix;
  prefix.before_task = 1;
  prefix.scales = Eigen::Vector3d(0.1,0.5,0.5);
  for (int rank : {1,3,0,1}) {
    high.active = rank != 0;
    if (rank == 3) {
      high.jacobian = Eigen::Matrix3d::Identity();
      high.desired_velocity = Eigen::Vector3d::Constant(3.0);
    } else {
      high.jacobian.resize(1,3); high.jacobian << 1,1,0;
      high.desired_velocity = Eigen::VectorXd::Constant(1,3.0);
    }
    const auto legacy = solver.solve({high,nominal});
    const auto extended = solver.solve({high,nominal},nullptr,1e-6,&prefix);
    ASSERT_TRUE(legacy.generalized_velocity.isApprox(extended.generalized_velocity));
    ASSERT_EQ(prefix.rank,static_cast<size_t>(rank));
    ASSERT_EQ(prefix.basis.cols(),3-rank);
    if (rank) {
      EXPECT_LT((high.jacobian*prefix.scales.asDiagonal()*prefix.basis).norm(),1e-12);
      EXPECT_TRUE((high.jacobian*prefix.particular).isApprox(high.desired_velocity));
    }
    EXPECT_LT((prefix.basis.transpose()*prefix.basis-Eigen::MatrixXd::Identity(3-rank,3-rank)).norm(),1e-12);
  }
}
