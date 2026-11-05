#pragma once

#include "task_priority_kinematic_control/core/common.hpp"

#include <Eigen/Core>

#include <vector>
#include <limits>
#include <string>

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

  struct Diagnostic
  {
    bool valid = true;
    std::string message;
    size_t failing_task = std::numeric_limits<size_t>::max();
    // Per priority: projected, weighted Jacobian, with structural zero rows removed.
    std::vector<double> minimum_singular_values;
    std::vector<bool> rank_deficient;
  };
  WholeBodyCommand solve(const std::vector<TaskComputation> & tasks,
    Diagnostic * diagnostic = nullptr, double singular_value_tolerance = 1e-6) const;
  const Eigen::VectorXd & velocity_limits() const {return velocity_limits_;}
  std::string configuration_key() const;
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
