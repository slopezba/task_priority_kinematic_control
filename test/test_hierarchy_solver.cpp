#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"

#include <gtest/gtest.h>

using task_priority_kinematic_control::HierarchySolver;
using task_priority_kinematic_control::SolverMethod;
using task_priority_kinematic_control::TaskComputation;

TEST(HierarchySolver, SolvesSingleTask)
{
  HierarchySolver solver;
  solver.configure(2);
  solver.set_method(SolverMethod::kDls);

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
