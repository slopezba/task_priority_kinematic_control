#pragma once

#include "task_priority_kinematic_control/core/common.hpp"

#include <Eigen/Core>

#include <vector>

namespace task_priority_kinematic_control
{

class HierarchySolver
{
public:
  void configure(size_t dofs);
  void set_method(SolverMethod method);
  void set_damping(double damping);
  void set_weights(const Eigen::VectorXd & weights);
  void set_velocity_limits(const Eigen::VectorXd & limits);

  WholeBodyCommand solve(const std::vector<TaskComputation> & tasks) const;
  double last_condition_estimate() const;

private:
  Eigen::MatrixXd weighted_inverse(const Eigen::MatrixXd & jacobian) const;
  Eigen::MatrixXd strict_pseudo_inverse(const Eigen::MatrixXd & matrix) const;
  Eigen::VectorXd apply_velocity_limits(const Eigen::VectorXd & command) const;

  size_t dofs_ = 0;
  SolverMethod method_ = SolverMethod::kDls;
  double damping_ = 0.05;
  Eigen::VectorXd weights_;
  Eigen::VectorXd velocity_limits_;
  mutable double last_condition_estimate_ = 0.0;
};

}  // namespace task_priority_kinematic_control
