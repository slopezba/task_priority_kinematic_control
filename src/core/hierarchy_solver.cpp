#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"

#include <Eigen/SVD>
#include <algorithm>

namespace task_priority_kinematic_control
{

void HierarchySolver::configure(size_t dofs)
{
  dofs_ = dofs;
  weights_ = Eigen::VectorXd::Ones(dofs_);
  velocity_limits_ = Eigen::VectorXd::Constant(dofs_, 1.0);
}

void HierarchySolver::set_method(SolverMethod method) { method_ = method; }
void HierarchySolver::set_damping(double damping) { damping_ = std::max(1e-6, damping); }
void HierarchySolver::set_weights(const Eigen::VectorXd & weights) { weights_ = weights; }
void HierarchySolver::set_velocity_limits(const Eigen::VectorXd & limits) { velocity_limits_ = limits; }
double HierarchySolver::last_condition_estimate() const { return last_condition_estimate_; }

WholeBodyCommand HierarchySolver::solve(const std::vector<TaskComputation> & tasks) const
{
  WholeBodyCommand out;
  out.generalized_velocity = Eigen::VectorXd::Zero(dofs_);
  Eigen::MatrixXd projector = Eigen::MatrixXd::Identity(dofs_, dofs_);

  for (const auto & task : tasks) {
    if (!task.active || task.jacobian.cols() != static_cast<Eigen::Index>(dofs_)) {
      continue;
    }
    const Eigen::MatrixXd augmented = task.jacobian * projector;
    if (augmented.rows() == 0 || augmented.cols() == 0) {
      continue;
    }
    const Eigen::MatrixXd inverse = weighted_inverse(augmented);
    out.generalized_velocity += inverse * (task.desired_velocity - task.jacobian * out.generalized_velocity);
    projector -= inverse * augmented;
  }

  out.generalized_velocity = apply_velocity_limits(out.generalized_velocity);
  return out;
}

Eigen::MatrixXd HierarchySolver::weighted_inverse(const Eigen::MatrixXd & jacobian) const
{
  const Eigen::VectorXd safe_weights = weights_.cwiseMax(1e-6);
  const Eigen::MatrixXd inv_w = safe_weights.cwiseInverse().asDiagonal();
  const Eigen::MatrixXd metric = jacobian * inv_w * jacobian.transpose();
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(metric, Eigen::ComputeThinU | Eigen::ComputeThinV);
  const Eigen::VectorXd singular_values = svd.singularValues();
  if (!singular_values.size()) {
    last_condition_estimate_ = 0.0;
  } else {
    const double smax = singular_values.maxCoeff();
    const double smin = std::max(1e-9, singular_values.minCoeff());
    last_condition_estimate_ = smax / smin;
  }

  switch (method_) {
    case SolverMethod::kPinv:
      return inv_w * jacobian.transpose() * metric.completeOrthogonalDecomposition().pseudoInverse();
    case SolverMethod::kSvd: {
      Eigen::JacobiSVD<Eigen::MatrixXd> jacobian_svd(
        jacobian * inv_w.cwiseSqrt(), Eigen::ComputeThinU | Eigen::ComputeThinV);
      Eigen::VectorXd inv_sigma = jacobian_svd.singularValues();
      for (Eigen::Index i = 0; i < inv_sigma.size(); ++i) {
        inv_sigma(i) = inv_sigma(i) > 1e-6 ? 1.0 / inv_sigma(i) : 0.0;
      }
      return inv_w.cwiseSqrt() * jacobian_svd.matrixV() * inv_sigma.asDiagonal() *
             jacobian_svd.matrixU().transpose();
    }
    case SolverMethod::kDls:
    default:
      return inv_w * jacobian.transpose() *
             (metric + damping_ * Eigen::MatrixXd::Identity(metric.rows(), metric.cols())).inverse();
  }
}

Eigen::VectorXd HierarchySolver::apply_velocity_limits(const Eigen::VectorXd & command) const
{
  Eigen::VectorXd limited = command;
  for (Eigen::Index i = 0; i < limited.size() && i < velocity_limits_.size(); ++i) {
    const double limit = std::max(1e-9, velocity_limits_(i));
    limited(i) = std::clamp(limited(i), -limit, limit);
  }
  return limited;
}

}  // namespace task_priority_kinematic_control
