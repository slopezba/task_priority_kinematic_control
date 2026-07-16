#include "task_priority_kinematic_control/core/hierarchy_solver.hpp"

#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <algorithm>
#include <cmath>
#include <limits>

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
    if (!task.active || task.jacobian.cols() != static_cast<Eigen::Index>(dofs_) ||
        task.desired_velocity.size() != task.jacobian.rows() || !task.jacobian.allFinite() ||
        !task.desired_velocity.allFinite())
    {
      continue;
    }
    const Eigen::MatrixXd augmented = task.jacobian * projector;
    if (augmented.rows() == 0 || augmented.cols() == 0 || !augmented.allFinite()) {
      continue;
    }
    const Eigen::MatrixXd inverse = weighted_inverse(augmented);
    if (inverse.rows() != static_cast<Eigen::Index>(dofs_) || inverse.cols() != augmented.rows() ||
        !inverse.allFinite())
    {
      continue;
    }
    const Eigen::VectorXd residual = task.desired_velocity - task.jacobian * out.generalized_velocity;
    const Eigen::VectorXd delta = projector * (inverse * residual);
    if (!delta.allFinite()) {
      continue;
    }
    out.generalized_velocity += delta;
    if (!out.generalized_velocity.allFinite()) {
      out.generalized_velocity.setZero();
      projector.setIdentity();
      continue;
    }
    const Eigen::MatrixXd strict_inverse = strict_pseudo_inverse(augmented);
    if (strict_inverse.rows() != static_cast<Eigen::Index>(dofs_) ||
        strict_inverse.cols() != augmented.rows() || !strict_inverse.allFinite())
    {
      continue;
    }
    projector -= projector * strict_inverse * augmented;
    if (!projector.allFinite()) {
      projector.setIdentity();
    }
  }

  out.generalized_velocity = apply_velocity_limits(out.generalized_velocity);
  return out;
}

Eigen::MatrixXd HierarchySolver::weighted_inverse(const Eigen::MatrixXd & jacobian) const
{
  if (jacobian.rows() == 0 || jacobian.cols() == 0 || !jacobian.allFinite()) {
    last_condition_estimate_ = 0.0;
    return Eigen::MatrixXd::Zero(jacobian.cols(), jacobian.rows());
  }

  Eigen::VectorXd safe_weights = Eigen::VectorXd::Ones(jacobian.cols());
  if (weights_.size() == jacobian.cols()) {
    safe_weights = weights_.cwiseMax(1e-6);
  }
  const Eigen::MatrixXd inv_w = safe_weights.cwiseInverse().asDiagonal();
  const Eigen::MatrixXd metric = jacobian * inv_w * jacobian.transpose();
  if (!metric.allFinite()) {
    last_condition_estimate_ = 0.0;
    return Eigen::MatrixXd::Zero(jacobian.cols(), jacobian.rows());
  }

  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(metric);
  if (eigensolver.info() == Eigen::Success && eigensolver.eigenvalues().allFinite()) {
    const Eigen::VectorXd eigenvalues = eigensolver.eigenvalues().cwiseAbs();
    const double smax = eigenvalues.maxCoeff();
    const double smin = std::max(1e-9, eigenvalues.minCoeff());
    last_condition_estimate_ = std::isfinite(smax) ? smax / smin : 0.0;
  } else {
    last_condition_estimate_ = 0.0;
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
    default: {
      const double damping_squared = damping_ * damping_;
      return inv_w * jacobian.transpose() *
             (metric + damping_squared * Eigen::MatrixXd::Identity(metric.rows(), metric.cols())).inverse();
    }
  }
}

Eigen::MatrixXd HierarchySolver::strict_pseudo_inverse(const Eigen::MatrixXd & matrix) const
{
  if (matrix.rows() == 0 || matrix.cols() == 0 || !matrix.allFinite()) {
    return Eigen::MatrixXd::Zero(matrix.cols(), matrix.rows());
  }

  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeThinU | Eigen::ComputeThinV);
  if (svd.singularValues().size() == 0 || !svd.singularValues().allFinite()) {
    return Eigen::MatrixXd::Zero(matrix.cols(), matrix.rows());
  }

  const double max_singular = svd.singularValues().maxCoeff();
  const double tolerance =
    std::max(matrix.rows(), matrix.cols()) * std::numeric_limits<double>::epsilon() * max_singular;
  Eigen::VectorXd inv_sigma = svd.singularValues();
  for (Eigen::Index i = 0; i < inv_sigma.size(); ++i) {
    inv_sigma(i) = inv_sigma(i) > tolerance ? 1.0 / inv_sigma(i) : 0.0;
  }

  return svd.matrixV() * inv_sigma.asDiagonal() * svd.matrixU().transpose();
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
