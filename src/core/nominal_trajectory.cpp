#include "task_priority_kinematic_control/core/nominal_trajectory.hpp"
#include "task_priority_kinematic_control/core/posture_scene.hpp"
#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"
#include "task_priority_kinematic_control/tasks/robot_nominal_task.hpp"
#include "sura_manipulation_actions/cartesian_pose_path.hpp"
#include "sura_manipulation_actions/whole_body_nominal.hpp"

#include <Eigen/Sparse>
#include <osqp.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <limits>
#include <optional>
#include <random>
#include <set>
#include <sstream>

namespace task_priority_kinematic_control
{
namespace
{
using V = Eigen::VectorXd;
using M = Eigen::MatrixXd;
using Vector6 = Eigen::Matrix<double, 6, 1>;
using Triplet = Eigen::Triplet<double>;
constexpr double kInfinity = 1e20;
constexpr double kTiltScale = 0.1;
constexpr double kSoftScale = 0.01;
// Numerical scale for constraint penalties, separate from equality acceptance.
constexpr double kHardTolerance = 1e-3;
constexpr double kMinimumClearance = 0.001;

// ----------------------------------------------------------------------------
// Sparse OSQP helper (the dense helper does not scale to all knots at once).
bool solve_sparse_qp(
  int n, const std::vector<Triplet> & p_entries, const V & q,
  int m, const std::vector<Triplet> & a_entries, const V & lower, const V & upper, V & solution)
{
  if (!q.allFinite() || !lower.allFinite() || !upper.allFinite()) {return false;}
  Eigen::SparseMatrix<double> full(n, n);
  full.setFromTriplets(p_entries.begin(), p_entries.end());
  Eigen::SparseMatrix<double> p = full.triangularView<Eigen::Upper>();
  for (int i = 0; i < n; ++i) {p.coeffRef(i, i) += 1e-9;}
  p.makeCompressed();
  Eigen::SparseMatrix<double> a(m, n);
  a.setFromTriplets(a_entries.begin(), a_entries.end());
  a.makeCompressed();
  const auto convert = [](const Eigen::SparseMatrix<double> & s, std::vector<c_float> & x,
      std::vector<c_int> & i, std::vector<c_int> & col, csc & out) {
      x.assign(s.valuePtr(), s.valuePtr() + s.nonZeros());
      i.assign(s.innerIndexPtr(), s.innerIndexPtr() + s.nonZeros());
      col.assign(s.outerIndexPtr(), s.outerIndexPtr() + s.cols() + 1);
      out.nzmax = static_cast<c_int>(x.size());
      out.m = static_cast<c_int>(s.rows()); out.n = static_cast<c_int>(s.cols());
      out.p = col.data(); out.i = i.data(); out.x = x.data(); out.nz = -1;
    };
  std::vector<c_float> px, ax, qv(q.data(), q.data() + n), lv(lower.data(), lower.data() + m),
  uv(upper.data(), upper.data() + m);
  std::vector<c_int> pi, pp, ai, ap;
  csc pm{}, am{};
  convert(p, px, pi, pp, pm);
  convert(a, ax, ai, ap, am);
  for (auto & value : lv) {value = std::max<c_float>(value, -OSQP_INFTY);}
  for (auto & value : uv) {value = std::min<c_float>(value, OSQP_INFTY);}
  OSQPData data{};
  data.n = n; data.m = m; data.P = &pm; data.A = &am;
  data.q = qv.data(); data.l = lv.data(); data.u = uv.data();
  OSQPSettings settings;
  osqp_set_default_settings(&settings);
  settings.verbose = 0;
  settings.polish = 1;
  settings.max_iter = 8000;
  settings.eps_abs = 1e-6;
  settings.eps_rel = 1e-6;
  OSQPWorkspace * workspace = nullptr;
  if (osqp_setup(&workspace, &data, &settings) != 0 || !workspace) {
    if (workspace) {osqp_cleanup(workspace);}
    return false;
  }
  const auto status = osqp_solve(workspace);
  const bool solved = status == 0 && workspace->info && workspace->solution && workspace->solution->x &&
    (workspace->info->status_val == OSQP_SOLVED ||
    workspace->info->status_val == OSQP_SOLVED_INACCURATE ||
    workspace->info->status_val == OSQP_MAX_ITER_REACHED);
  if (solved) {
    solution.resize(n);
    for (int i = 0; i < n; ++i) {solution(i) = workspace->solution->x[i];}
  }
  osqp_cleanup(workspace);
  return solved && solution.allFinite();
}

// Quadratic cost builder: minimizes 1/2 x'Px + q'x.
struct Cost
{
  std::vector<Triplet> p;
  V q;
  explicit Cost(int n) : q(V::Zero(n)) {}
  // w * (c + sum a_i x_i)^2
  void square(const std::vector<std::pair<int, double>> & terms, double c, double w)
  {
    if (w <= 0.0) {return;}
    for (const auto & [i, ai] : terms) {
      if (ai == 0.0) {continue;}
      q(i) += 2.0 * w * c * ai;
      for (const auto & [j, aj] : terms) {if (aj != 0.0) {p.emplace_back(i, j, 2.0 * w * ai * aj);}}
    }
  }
};

struct Constraints
{
  std::vector<Triplet> a;
  std::vector<double> lower, upper;
  void row(const std::vector<std::pair<int, double>> & terms, double l, double u)
  {
    const int r = static_cast<int>(lower.size());
    for (const auto & [i, value] : terms) {if (value != 0.0) {a.emplace_back(r, i, value);}}
    lower.push_back(std::min(l, u)); upper.push_back(std::max(l, u));
  }
};

// ω = E(rpy) * d(rpy)/dt for ZYX Euler coordinates (as in the posture search).
Eigen::Matrix3d spatial(const Eigen::Vector3d & rpy)
{
  Eigen::Matrix3d out;
  const Eigen::AngleAxisd yaw(rpy.z(), Eigen::Vector3d::UnitZ());
  const Eigen::AngleAxisd pitch(rpy.y(), Eigen::Vector3d::UnitY());
  out.col(0) = yaw * pitch * Eigen::Vector3d::UnitX();
  out.col(1) = yaw * Eigen::Vector3d::UnitY();
  out.col(2) = Eigen::Vector3d::UnitZ();
  return out;
}

// Allowed new error e' = e - jΔ in [lo, hi].
struct Row
{
  Eigen::RowVectorXd j;
  double e = 0.0, lo = 0.0, hi = 0.0;
};

struct Knot
{
  M tip_j;
  Vector6 tip_e = Vector6::Zero();
  std::vector<Row> hard, soft;
  std::vector<PreventiveMargin> preventive;
  double clearance = std::numeric_limits<double>::infinity();
  V clearance_gradient;
  bool safe = true;
  std::string failure;
};

struct Evaluation
{
  std::vector<Knot> knots;  // index 0 is the fixed start
  double cost = std::numeric_limits<double>::infinity();
  double violation = std::numeric_limits<double>::infinity();
  std::string worst;
  double prevention = 0.0;
  double merit(double mu) const {return cost + mu * violation;}
};

class Planner
{
public:
  Planner(const StpikSnapshot & start, const Preparation::Goal & goal, const StpikOptions & simulation,
    const PostureOptions & posture, const NominalOptions & options, const StpikCheck::Canceled & canceled,
    const NominalFeedback & feedback, bool free_start)
  : start_(start), goal_(goal), simulation_(simulation), posture_(posture), options_(options),
    canceled_(canceled), feedback_(feedback), scene_(goal, *start.model),
    deadline_(std::chrono::steady_clock::now() +
      std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(options.budget)))
  {
    free_start_ = free_start;
    if (!start.unavailable_reason.empty() || !start.model || !start.backend) {
      throw std::runtime_error("Nominal planning snapshot unavailable: " + start.unavailable_reason);
    }
    const auto & model = *start.model;
    if (model.active_base_dofs() != std::vector<int>({0, 1, 2, 3, 4, 5})) {
      throw std::runtime_error("Nominal planning requires all six base DOFs");
    }
    joints_ = static_cast<int>(model.all_joint_names().size());
    n_ = 6 + joints_;
    if (!StpikTaskAccess::joint_bounds(start.tasks, lower_, upper_) || lower_.size() != joints_) {
      throw std::runtime_error("Nominal planning requires enabled finite joint limits");
    }
    if (!StpikTaskAccess::tilt_bounds(start.tasks, tilt_lower_, tilt_upper_)) {
      tilt_lower_ = Eigen::Vector2d::Constant(-1.2); tilt_upper_ = Eigen::Vector2d::Constant(1.2);
    }
    velocity_ = start.solver.velocity_limits();
    if (velocity_.size() != n_ || !velocity_.allFinite() || (velocity_.array() <= 0.0).any()) {
      throw std::runtime_error("Nominal planning needs finite positive generalized velocity limits");
    }
    if (!simulation.velocity_limits.empty()) {
      if (simulation.velocity_limits.size() != static_cast<size_t>(n_)) {
        throw std::runtime_error("stpik.velocity_limits does not match the generalized order");
      }
      velocity_ = velocity_.cwiseMin(Eigen::Map<const V>(simulation.velocity_limits.data(), n_));
    }
    velocity_ *= options.velocity_fraction;
    scale_ = V::Constant(n_, 0.5);
    scale_.head<3>().setConstant(0.1);
    scale_.segment<3>(3).setConstant(0.2);
    std::string reason;
    if (goal.trajectory.header.frame_id != model.world_frame() || !path_.initTimed(goal.trajectory, reason)) {
      throw std::runtime_error("Invalid nominal tool path: " + (reason.empty() ? "frame mismatch" : reason));
    }
    const double lin = *std::min_element(goal.path_tolerance.begin(), goal.path_tolerance.begin() + 3);
    const double ang = *std::min_element(goal.path_tolerance.begin() + 3, goal.path_tolerance.end());
    band_ << Eigen::Vector3d::Constant(std::max(1e-4, 0.25 * lin)), Eigen::Vector3d::Constant(std::max(1e-4, 0.25 * ang));
    const double duration = std::max(path_.duration(), 0.2);
    const int count = std::clamp(static_cast<int>(std::ceil(duration / options.knot_dt)), 2,
      std::max(2, static_cast<int>(options.max_knots)));
    for (int k = 0; k <= count; ++k) {times_.push_back(duration * k / count);}
    dt_ = duration / count;
    total_ = duration;
    current_ = coords(start.initial_state);
    // Free start: the repositioning cost uses the posture search scales, and the move must
    // fit the posture duration with the quintic profile (peak speed 1.875 * distance / T).
    move_scale_ = V::Ones(n_);
    move_scale_.head<3>().setConstant(1.0 / std::max(posture.radius, 1e-6));
    move_scale_.tail(joints_) = (upper_ - lower_).cwiseInverse();
    max_move_ = velocity_ * posture.duration / 1.875;
    classify();
  }

  NominalPlan plan(const NominalPlan * warm)
  {
    // A free start whose repositioning cannot settle is retried closer to the measured
    // posture, and finally with the start fixed at it.
    std::string rejected;
    for (int attempt = 0;; ++attempt) {
      start_rejected_ = false;
      try {
        return plan_once(warm);
      } catch (const std::runtime_error & error) {
        if (!free_start_ || !start_rejected_ || expired()) {
          if (rejected.empty()) {throw;}
          throw std::runtime_error(std::string(error.what()) + "; earlier free-start attempts: " + rejected);
        }
        rejected += (rejected.empty() ? "" : " | ") + std::string(error.what());
        if (attempt < 2) {
          max_move_ *= 0.4;
          report("Nominal: start posture could not settle; retrying closer to the measured posture");
        } else {
          free_start_ = false;
          report("Nominal: retrying with the start fixed at the measured posture");
        }
      }
    }
  }

  NominalPlan plan_once(const NominalPlan * warm)
  {
    std::vector<V> seed;
    std::string seed_method;
    if (warm && !warm->knots.empty()) {
      seed = from_warm(*warm); seed_method = "warm_start";
    } else if (options_.rollout_seed) {
      report("Nominal: STPIK rollout seed");
      seed_method = "stpik_rollout";
      if (!rollout(seed)) {
        // Keep the reached part and project the remainder onto the tool path; the SQP
        // can then repair it (e.g. push a colliding segment out with the signed distance).
        seed_method = "stpik_rollout_repaired";
        if (!repair(seed)) {seed.clear();}
      }
    }
    std::string failure;
    if (!seed.empty()) {
      report("Nominal: SQP refinement (" + seed_method + ")");
      auto refined = sqp(seed, failure);
      if (!refined.empty()) {
        auto plan = finish(refined, seed_method + "+sqp", failure);
        if (plan) {return *plan;}
      }
      // The rollout itself is the controller's own behaviour: keep it if it validates.
      if (seed_method == "stpik_rollout" && !free_start_) {
        auto plan = finish(seed, seed_method, failure);
        if (plan) {return *plan;}
      }
    }
    if (options_.global_fallback && !expired()) {
      report("Nominal: layered self-motion sampling + dynamic programming");
      auto global = layered(seed, failure);
      if (!global.empty()) {
        report("Nominal: SQP refinement (layered_dp)");
        std::string sqp_failure;
        auto refined = sqp(global, sqp_failure);
        if (!refined.empty()) {
          auto plan = finish(refined, "layered_dp+sqp", failure);
          if (plan) {return *plan;}
        }
        auto plan = finish(global, "layered_dp", failure);
        if (plan) {return *plan;}
      }
    }
    if (canceled_()) {throw std::runtime_error("Nominal trajectory planning canceled");}
    throw std::runtime_error("Nominal trajectory planning failed" +
      std::string(expired() ? " (budget exhausted)" : "") + ": " + failure);
  }

private:
  // --------------------------------------------------------------------------
  // Setup
  void classify()
  {
    auto probe = clone();
    for (size_t i = 0; i < probe.tasks.size(); ++i) {
      if (probe.tasks[i]->id() == goal_.task_id) {selected_ = static_cast<int>(i);}
      if (std::dynamic_pointer_cast<RobotNominalTask>(probe.tasks[i])) {nominal_ = static_cast<int>(i);}
    }
    if (selected_ < 0) {throw std::runtime_error("Selected tool task is unavailable: " + goal_.task_id);}
    if (nominal_ < 0) {throw std::runtime_error("No RobotNominalTask (robot_nominal) is configured");}
    const auto robot = std::dynamic_pointer_cast<RobotNominalTask>(probe.tasks[static_cast<size_t>(nominal_)]);
    if (!robot->enabled()) {throw std::runtime_error("robot_nominal task is disabled");}
    if (nominal_ < selected_) {throw std::runtime_error("robot_nominal must have lower priority than " + goal_.task_id);}
    std::set<std::string> excluded(robot->sync_tasks().begin(), robot->sync_tasks().end());
    excluded.insert({"base_pose", "left_arm_nominal", "right_arm_nominal", goal_.task_id});
    for (int i = 0; i < nominal_; ++i) {
      if (excluded.count(probe.tasks[static_cast<size_t>(i)]->id())) {continue;}
      considered_.push_back({i, i < selected_});
    }
  }
  StpikSnapshot clone() const
  {
    auto out = StpikTaskAccess::capture(start_.initial_state, start_.model, start_.backend, start_.solver, start_.tasks);
    if (!out.unavailable_reason.empty()) {throw std::runtime_error(out.unavailable_reason);}
    out.captured = start_.captured;
    return out;
  }
  bool expired() const {return canceled_() || std::chrono::steady_clock::now() >= deadline_;}
  void report(const std::string & message) const {if (feedback_) {feedback_(message);}}
  V coords(const WholeBodyState & s) const
  {
    V x(n_);
    x << s.base_position, s.base_rpy, s.joint_positions;
    return x;
  }
  WholeBodyState state(const V & x) const
  {
    auto s = start_.initial_state;
    s.base_position = x.head<3>(); s.base_rpy = x.segment<3>(3);
    s.joint_positions = x.tail(joints_);
    s.joint_velocities = V::Zero(joints_);
    return s;
  }
  M to_coordinates(M j, const V & x) const
  {
    if (j.cols() != n_) {throw std::runtime_error("Unexpected task Jacobian width");}
    j.middleCols(3, 3) = j.middleCols(3, 3) * spatial(x.segment<3>(3));
    return j;
  }
  Vector6 tip_error(const Eigen::Isometry3d & desired, const Eigen::Isometry3d & measured) const
  {
    Vector6 e;
    const Eigen::AngleAxisd rotation(desired.linear() * measured.linear().transpose());
    e << desired.translation() - measured.translation(), rotation.axis() * rotation.angle();
    return e;
  }

  // --------------------------------------------------------------------------
  // Seeds
  bool rollout(std::vector<V> & seed)
  {
    seed.assign(times_.size(), V());
    seed[0] = coords(start_.initial_state);
    size_t next = 1;
    auto sim = clone();
    auto check = posture_tracking_goal(goal_);
    const auto result = StpikCheck::run(std::move(sim), check, simulation_,
      [this] {return expired();}, {}, [&](const WholeBodyState & s, double time) {
        while (next < times_.size() && time + 1e-9 >= times_[next]) {seed[next++] = coords(s);}
      });
    // Hold the last reached configuration for knots past the path duration.
    for (size_t k = 1; k < seed.size(); ++k) {
      if (seed[k].size() == 0 && times_[k] > path_.duration() + 1e-9 && seed[k - 1].size()) {seed[k] = seed[k - 1];}
    }
    rollout_ = seed;
    const bool complete = std::all_of(seed.begin(), seed.end(), [](const V & x) {return x.size() > 0;});
    if (result.outcome != result.SUCCESS || !complete) {
      rollout_failure_ = result.code + ": " + result.message;
      report("Nominal: rollout seed failed (" + result.code + ")");
      return false;
    }
    return true;
  }
  std::vector<V> from_warm(const NominalPlan & warm) const
  {
    std::vector<V> out;
    const V start = coords(start_.initial_state);
    const V offset = start - interpolate(warm.times, warm.knots, 0.0);
    for (double t : times_) {
      const double fade = std::clamp(1.0 - t / std::max(0.25 * total_, 1e-6), 0.0, 1.0);
      out.push_back(interpolate(warm.times, warm.knots, t) + fade * offset);
    }
    out.front() = start;
    return out;
  }
  static V interpolate(const std::vector<double> & times, const std::vector<V> & knots, double t)
  {
    if (t <= times.front()) {return knots.front();}
    if (t >= times.back()) {return knots.back();}
    const auto after = std::upper_bound(times.begin(), times.end(), t);
    const size_t b = static_cast<size_t>(after - times.begin()), a = b - 1;
    const double u = (t - times[a]) / (times[b] - times[a]);
    return (1.0 - u) * knots[a] + u * knots[b];
  }

  // --------------------------------------------------------------------------
  // Evaluation and linearization at every knot.
  Evaluation evaluate(const std::vector<V> & x, bool gradients)
  {
    Evaluation out;
    out.knots.resize(x.size());
    out.cost = 0.0; out.violation = 0.0;
    auto probe = clone();
    double worst = 0.0;
    const auto violate = [&](double amount, const std::string & what) {
        if (amount <= 0.0) {return;}
        out.violation += amount;
        if (amount > worst) {worst = amount; out.worst = what;}
      };
    for (size_t k = 0; k < x.size(); ++k) {
      auto & knot = out.knots[k];
      const auto s = state(x[k]);
      probe.backend->update(s);
      std::vector<TaskComputation> computations(static_cast<size_t>(nominal_));
      for (const auto & [index, hard] : considered_) {
        const auto & task = probe.tasks[static_cast<size_t>(index)];
        if (task->enabled()) {computations[static_cast<size_t>(index)] = task->update(s, *probe.backend);}
      }
      if (k == 0 && !free_start_) {continue;}  // Fixed start: only initialize latches with it.
      const double w = dt_ / total_;
      const auto at = "t=" + std::to_string(times_[k]) + " s";
      // Tool pose on the timed path.
      const auto frame = pose_math::compute_tip_state(*probe.backend, s, goal_.tip_frame, probe.tasks[static_cast<size_t>(selected_)].get());
      knot.tip_j = to_coordinates(frame.jacobian, x[k]);
      knot.tip_e = tip_error(path_.sample(times_[k]), frame.pose);
      for (int i = 0; i < 6; ++i) {
        violate((std::abs(knot.tip_e(i)) - band_(i)) / band_(i), "tool path " + at);
        out.cost += 1e3 * w * std::pow(knot.tip_e(i) / band_(i), 2) * (std::abs(knot.tip_e(i)) > band_(i));
      }
      // Higher tasks.
      for (const auto & [index, hard] : considered_) {
        const auto & task = *probe.tasks[static_cast<size_t>(index)];
        if (!task.enabled()) {continue;}
        StpikAction::Result safety;
        if (!StpikTaskAccess::safety(task, s, simulation_.numerical_tolerance, safety)) {
          knot.safe = false; knot.failure = safety.code + "/" + safety.constraint;
          violate(1.0, task.id() + " safety " + at);
        }
        for (auto margin : StpikTaskAccess::preventive_margins(task, s)) {
          M gradient = margin.gradient;
          margin.gradient = to_coordinates(gradient, x[k]);
          out.prevention += w * std::pow(std::max(0.0, margin.deficit), 2);
          knot.preventive.push_back(std::move(margin));
        }
        PostureRegion region;
        if (StpikTaskAccess::posture_region(task, region, true)) {
          region.jacobian = to_coordinates(region.jacobian, x[k]);
          const double scale = region.radial ? std::sqrt(std::max<double>(1.0, (region.jacobian.rowwise().norm().array() > 0).count())) : 1.0;
          for (Eigen::Index r = 0; r < region.error.size(); ++r) {
            if (region.jacobian.row(r).isZero(1e-12)) {continue;}
            const double limit = region.always_active ? options_.equality_tolerance : region.limit(region.radial ? 0 : r) / scale;
            add_hard(knot, out, region.jacobian.row(r), region.error(r), -limit, limit, task.id() + " " + at, violate);
          }
          continue;
        }
        const auto & computation = computations[static_cast<size_t>(index)];
        if (!computation.active || computation.jacobian.rows() == 0) {continue;}
        const M j = to_coordinates(computation.jacobian, x[k]);
        for (Eigen::Index r = 0; r < j.rows(); ++r) {
          if (j.row(r).isZero(1e-12)) {continue;}
          const double e = computation.error(r);
          if (computation.structural_zero_rows) {
            if (e == 0.0) {continue;}
            add_hard(knot, out, j.row(r), e, e > 0.0 ? -kInfinity : e, e < 0.0 ? kInfinity : e,
              task.id() + " " + at, violate);
          } else if (hard) {
            add_hard(knot, out, j.row(r), e, -options_.equality_tolerance, options_.equality_tolerance,
              task.id() + " " + at, violate);
          } else {
            const double residual = e - std::clamp(e, -options_.equality_tolerance, options_.equality_tolerance);
            knot.soft.push_back({j.row(r), residual, 0.0, 0.0});
            out.cost += options_.soft_task_weight * w * std::pow(residual / kSoftScale, 2);
          }
        }
      }
      // Bounds.
      for (int j = 0; j < joints_; ++j) {
        const double q = x[k](6 + j);
        violate((lower_(j) - q) / 0.01, "joint lower bound " + at);
        violate((q - upper_(j)) / 0.01, "joint upper bound " + at);
        const double low = q - lower_(j), high = upper_(j) - q;
        const double buffer = posture_.joint_buffer;
        if (low < buffer) {out.cost += posture_.joints * w * std::pow((buffer - low) / buffer, 2);}
        if (high < buffer) {out.cost += posture_.joints * w * std::pow((buffer - high) / buffer, 2);}
      }
      for (int i = 0; i < 2; ++i) {
        const double angle = x[k](3 + i);
        violate((tilt_lower_(i) - angle) / 0.01, "tilt bound " + at);
        violate((angle - tilt_upper_(i)) / 0.01, "tilt bound " + at);
        out.cost += posture_.tilt * w * std::pow(angle / kTiltScale, 2);
      }
      // MoveIt clearance.
      knot.clearance = distance(s);
      violate((kMinimumClearance - knot.clearance) / kMinimumClearance, "MoveIt clearance " + at);
      if (knot.clearance < posture_.collision_buffer) {
        out.cost += posture_.clearance * w * std::pow((posture_.collision_buffer - knot.clearance) /
          posture_.collision_buffer, 2);
        if (gradients) {
          knot.clearance_gradient = V::Zero(n_);
          for (int c = 0; c < n_; ++c) {
            V perturbed = x[k];
            perturbed(c) += 1e-4;
            knot.clearance_gradient(c) = (distance(state(perturbed)) - knot.clearance) / 1e-4;
          }
        }
      }
      if (k == 0) {
        // Free start: cost of moving from the measured posture, within a feasible move.
        const V move = x[0] - current_;
        out.cost += posture_.movement * move.cwiseProduct(move_scale_).squaredNorm();
        for (int c = 0; c < n_; ++c) {
          violate((std::abs(move(c)) - max_move_(c)) / max_move_(c), "repositioning distance");
        }
        continue;
      }
      // Velocity between knots.
      const V difference = x[k] - x[k - 1];
      for (int c = 0; c < n_; ++c) {
        const double limit = velocity_(c) * dt_;
        violate((std::abs(difference(c)) - limit) / limit, "velocity " + at);
      }
    }
    // Smoothness (accelerations at rest at both ends).
    const int last = static_cast<int>(x.size()) - 1;
    for (int k = 0; k <= last; ++k) {
      const V acceleration = (x[static_cast<size_t>(k == last ? last : k + 1)] - 2.0 * x[static_cast<size_t>(k)] +
        x[static_cast<size_t>(k == 0 ? 0 : k - 1)]) / (dt_ * dt_);
      out.cost += options_.smooth_weight * dt_ / total_ * acceleration.cwiseQuotient(scale_).squaredNorm();
    }
    return out;
  }
  // Positive clearance, or negative penetration depth inside a collision.
  double distance(const WholeBodyState & s)
  {
    const double clearance = scene_.clearance(s);
    if (clearance > 0.0) {return clearance;}
    return std::min(scene_.signed_clearance(s), 0.0);
  }
  // Complete a partial seed by projecting each missing knot from the previous one.
  bool repair(std::vector<V> & seed)
  {
    auto probe = clone();
    for (size_t k = 1; k < seed.size() && !expired(); ++k) {
      if (seed[k].size()) {continue;}
      V x = seed[k - 1];
      if (!project(x, times_[k], probe)) {
        report("Nominal: seed projection failed at t=" + std::to_string(times_[k]) + " s");
        return false;
      }
      seed[k] = x;
    }
    return !expired() && std::all_of(seed.begin(), seed.end(), [](const V & x) {return x.size() > 0;});
  }
  template<class Violate>
  void add_hard(Knot & knot, Evaluation & out, const Eigen::RowVectorXd & j, double e, double lo, double hi,
    const std::string & what, const Violate & violate) const
  {
    knot.hard.push_back({j, e, lo, hi});
    const double excess = std::max(lo - e, e - hi);
    violate(excess / kHardTolerance, what);
    if (excess > 0.0) {out.cost += 1e3 * std::pow(excess / kHardTolerance, 2) * dt_ / total_;}
  }

  // --------------------------------------------------------------------------
  // One SQP step: QP in the increments of knots 1..N.
  bool step(const std::vector<V> & x, const Evaluation & e, const V & trust, std::vector<V> & candidate) const
  {
    const int knots = static_cast<int>(x.size()) - 1;
    const int first = free_start_ ? 0 : 1;
    const int vars = (knots + 1 - first) * n_;
    const auto var = [&](int k, int c) {return (k - first) * n_ + c;};
    const double w = dt_ / total_;
    Cost cost(vars);
    Constraints rows;
    // Smoothness.
    for (int k = 0; k <= knots; ++k) {
      const int prev = k == 0 ? 0 : k - 1, next = k == knots ? knots : k + 1;
      std::vector<std::pair<int, double>> coefficients{{prev, 1.0}, {k, -2.0}, {next, 1.0}};
      for (int c = 0; c < n_; ++c) {
        const double factor = 1.0 / (dt_ * dt_ * scale_(c));
        double constant = 0.0;
        std::vector<std::pair<int, double>> terms;
        for (const auto & [index, coefficient] : coefficients) {
          constant += coefficient * x[static_cast<size_t>(index)](c) * factor;
          if (index >= first) {terms.push_back({var(index, c), coefficient * factor});}
        }
        cost.square(terms, constant, options_.smooth_weight * w);
      }
    }
    for (int k = first; k <= knots; ++k) {
      const auto & knot = e.knots[static_cast<size_t>(k)];
      const V & xk = x[static_cast<size_t>(k)];
      const auto terms_of = [&](const Eigen::RowVectorXd & j, double factor) {
          std::vector<std::pair<int, double>> terms;
          for (int c = 0; c < n_; ++c) {if (j(c) != 0.0) {terms.push_back({var(k, c), j(c) * factor});}}
          return terms;
        };
      // Proximity regularization and variable bounds (trust region, joints, tilt).
      for (int c = 0; c < n_; ++c) {
        cost.square({{var(k, c), 1.0 / scale_(c)}}, 0.0, 1e-4);
        double lo = -trust(c), hi = trust(c);
        if (c >= 6) {lo = std::max(lo, lower_(c - 6) - xk(c)); hi = std::min(hi, upper_(c - 6) - xk(c));}
        if (c == 3 || c == 4) {lo = std::max(lo, tilt_lower_(c - 3) - xk(c)); hi = std::min(hi, tilt_upper_(c - 3) - xk(c));}
        if (k == 0) {
          lo = std::max(lo, current_(c) - max_move_(c) - xk(c));
          hi = std::min(hi, current_(c) + max_move_(c) - xk(c));
          const double move = (xk(c) - current_(c)) * move_scale_(c);
          cost.square({{var(k, c), move_scale_(c)}}, move, posture_.movement);
        }
        rows.row({{var(k, c), 1.0}}, std::min(lo, 0.0), std::max(hi, 0.0));
      }
      // Tilt and joint margin costs.
      for (int i = 0; i < 2; ++i) {
        cost.square({{var(k, 3 + i), 1.0 / kTiltScale}}, xk(3 + i) / kTiltScale, posture_.tilt * w);
      }
      for (int j = 0; j < joints_; ++j) {
        const double buffer = posture_.joint_buffer;
        const double low = xk(6 + j) - lower_(j), high = upper_(j) - xk(6 + j);
        if (low < buffer) {cost.square({{var(k, 6 + j), -1.0 / buffer}}, (buffer - low) / buffer, posture_.joints * w);}
        if (high < buffer) {cost.square({{var(k, 6 + j), 1.0 / buffer}}, (buffer - high) / buffer, posture_.joints * w);}
      }
      // Tool path: never worse than now, pulled strongly into the band.
      for (int i = 0; i < 6; ++i) {
        const Eigen::RowVectorXd j = knot.tip_j.row(i);
        const double err = knot.tip_e(i), lo = std::min(-band_(i), err), hi = std::max(band_(i), err);
        rows.row(terms_of(j, 1.0), err - hi, err - lo);
        cost.square(terms_of(j, -1.0 / band_(i)), err / band_(i), 1e3 * w);
      }
      for (const auto & row : knot.hard) {
        const double lo = std::min(row.lo, row.e), hi = std::max(row.hi, row.e);
        rows.row(terms_of(row.j, 1.0), row.e - hi, row.e - lo);
        const double target = row.e - std::clamp(row.e, row.lo, row.hi);
        if (target != 0.0) {cost.square(terms_of(row.j, 1.0 / kHardTolerance), -target / kHardTolerance, 1e3 * w);}
      }
      for (const auto & row : knot.soft) {
        cost.square(terms_of(row.j, -1.0 / kSoftScale), row.e / kSoftScale, options_.soft_task_weight * w);
      }
      if (knot.clearance_gradient.size() == n_ && knot.clearance < posture_.collision_buffer) {
        const Eigen::RowVectorXd g = knot.clearance_gradient.transpose();
        rows.row(terms_of(g, 1.0), std::min(kMinimumClearance - knot.clearance, 0.0), kInfinity);
        cost.square(terms_of(g, -1.0 / posture_.collision_buffer),
          (posture_.collision_buffer - knot.clearance) / posture_.collision_buffer, posture_.clearance * w);
      }
      // Velocity limits between consecutive knots; never forbid the current difference.
      if (k == 0) {continue;}
      const V difference = xk - x[static_cast<size_t>(k - 1)];
      for (int c = 0; c < n_; ++c) {
        const double limit = velocity_(c) * dt_;
        const double lo = std::min(-limit, difference(c)), hi = std::max(limit, difference(c));
        std::vector<std::pair<int, double>> terms{{var(k, c), 1.0}};
        if (k - 1 >= first) {terms.push_back({var(k - 1, c), -1.0});}
        rows.row(terms, lo - difference(c), hi - difference(c));
      }
    }
    std::vector<std::pair<std::vector<std::pair<int, double>>, double>> preventive;
    if (e.violation <= 1e-3) {
      for (int k = first; k <= knots; ++k) {
        for (const auto & margin : e.knots[static_cast<size_t>(k)].preventive) {
          std::vector<std::pair<int, double>> terms;
          for (int c = 0; c < n_; ++c) {
            if (margin.gradient(c) != 0.0) {terms.push_back({var(k, c), margin.gradient(c)});}
          }
          preventive.push_back({std::move(terms), margin.deficit});
        }
      }
    }
    const int variables = vars + static_cast<int>(preventive.size());
    const auto solve = [&](const Cost & objective, V & solution) {
        const V lower = Eigen::Map<const V>(rows.lower.data(), rows.lower.size());
        const V upper = Eigen::Map<const V>(rows.upper.data(), rows.upper.size());
        if (!solve_sparse_qp(variables, objective.p, objective.q,
          static_cast<int>(lower.size()), rows.a, lower, upper, solution)) {return false;}
        // OSQP's iteration-limit status alone does not establish feasibility.
        V values = V::Zero(lower.size());
        for (const auto & entry : rows.a) {values(entry.row()) += entry.value() * solution(entry.col());}
        return (values.array() >= lower.array() - 5e-5).all() &&
               (values.array() <= upper.array() + 5e-5).all();
      };
    V solution;
    if (!preventive.empty()) {
      Cost primary(variables);
      for (size_t i = 0; i < preventive.size(); ++i) {
        const int slack = vars + static_cast<int>(i);
        auto terms = preventive[i].first;
        terms.push_back({slack, -1.0});
        rows.row(terms, -kInfinity, -preventive[i].second);
        rows.row({{slack, 1.0}}, 0.0, kInfinity);
        primary.square({{slack, 1.0}}, 0.0, w);
      }
      if (!solve(primary, solution)) {return false;}
      // Preserve each optimal deficit. This conservative hierarchy stays a QP;
      // an exact bound on their squared sum would require a quadratic constraint.
      for (size_t i = 0; i < preventive.size(); ++i) {
        const int slack = vars + static_cast<int>(i);
        rows.row({{slack, 1.0}}, 0.0, std::max(0.0, solution(slack)) + 1e-5);
      }
      cost.q.conservativeResize(variables);
      cost.q.tail(static_cast<Eigen::Index>(preventive.size())).setZero();
    }
    if (!solve(cost, solution)) {return false;}
    candidate = x;
    for (int k = first; k <= knots; ++k) {candidate[static_cast<size_t>(k)] += solution.segment(var(k, 0), n_);}
    return true;
  }

  std::vector<V> sqp(std::vector<V> x, std::string & failure)
  {
    V trust(n_);
    trust.head<3>().setConstant(options_.trust_position);
    trust.segment<3>(3).setConstant(options_.trust_angle);
    trust.tail(joints_).setConstant(options_.trust_joint);
    constexpr double mu = 100.0;
    auto current = evaluate(x, true);
    const double initial_prevention = current.prevention;
    for (int iteration = 0; iteration < static_cast<int>(options_.iterations) && !expired(); ++iteration) {
      std::vector<V> candidate;
      if (!step(x, current, trust, candidate)) {
        trust *= 0.5;
        if (trust.maxCoeff() < 1e-3) {break;}
        continue;
      }
      bool accepted = false;
      double largest = 0.0;
      for (double alpha : {1.0, 0.5, 0.25}) {
        std::vector<V> trial = x;
        largest = 0.0;
        for (size_t k = free_start_ ? 0 : 1; k < x.size(); ++k) {
          const V delta = alpha * (candidate[k] - x[k]);
          trial[k] += delta;
          largest = std::max(largest, delta.cwiseAbs().maxCoeff());
        }
        auto evaluation = evaluate(trial, true);
        // Recover feasibility first. Once feasible, prevention precedes all
        // secondary costs, including during nonlinear line search.
        const bool improves = current.violation > 1e-3 ?
          evaluation.merit(mu) < current.merit(mu) - 1e-9 :
          evaluation.violation <= 1e-3 &&
          (evaluation.prevention < current.prevention - 1e-7 ||
          (evaluation.prevention <= current.prevention + 1e-7 &&
          evaluation.cost < current.cost - 1e-9));
        if (improves) {
          x = std::move(trial); current = std::move(evaluation); accepted = true; break;
        }
        if (expired()) {break;}
      }
      if (!accepted) {
        trust *= 0.5;
        if (trust.maxCoeff() < 1e-3) {break;}
        continue;
      }
      if (largest < 1e-4) {break;}
    }
    if (current.violation > 1e-3) {
      failure = "SQP left a constraint violation (" + current.worst + ")";
      return {};
    }
    last_cost_ = current.cost;
    report("Nominal: preventive cost " + std::to_string(initial_prevention) + " -> " +
      std::to_string(current.prevention) + "; secondary cost=" + std::to_string(current.cost));
    return x;
  }

  // --------------------------------------------------------------------------
  // Global fallback: layered self-motion samples + dynamic programming.
  bool project(V & x, double time, StpikSnapshot & probe) const
  {
    const auto desired = path_.sample(time);
    for (int iteration = 0; iteration < 40; ++iteration) {
      const auto s = state(x);
      probe.backend->update(s);
      const auto frame = pose_math::compute_tip_state(*probe.backend, s, goal_.tip_frame, probe.tasks[static_cast<size_t>(selected_)].get());
      std::vector<Eigen::RowVectorXd> rows;
      std::vector<double> errors;
      const Vector6 tip = tip_error(desired, frame.pose);
      const M tip_j = to_coordinates(frame.jacobian, x);
      bool converged = true;
      for (int i = 0; i < 6; ++i) {
        rows.push_back(tip_j.row(i)); errors.push_back(tip(i));
        converged = converged && std::abs(tip(i)) <= band_(i);
      }
      for (const auto & [index, hard] : considered_) {
        auto & task = *probe.tasks[static_cast<size_t>(index)];
        if (!task.enabled()) {continue;}
        const auto computation = task.update(s, *probe.backend);
        PostureRegion region;
        if (StpikTaskAccess::posture_region(task, region) || !computation.active ||
          computation.structural_zero_rows || computation.jacobian.rows() == 0) {continue;}
        const M j = to_coordinates(computation.jacobian, x);
        for (Eigen::Index r = 0; r < j.rows(); ++r) {
          if (j.row(r).isZero(1e-12)) {continue;}
          const double error = computation.error(r);
          if (std::abs(error) <= options_.equality_tolerance) {continue;}
          // Aim inside the band so damped projection can reach it in finite iterations.
          const double residual = error - std::copysign(0.5 * options_.equality_tolerance, error);
          rows.push_back(j.row(r)); errors.push_back(residual);
          converged = false;
        }
      }
      if (converged) {return true;}
      M j(static_cast<Eigen::Index>(rows.size()), n_);
      V e(static_cast<Eigen::Index>(rows.size()));
      for (size_t r = 0; r < rows.size(); ++r) {
        j.row(static_cast<Eigen::Index>(r)) = rows[r];
        e(static_cast<Eigen::Index>(r)) = errors[r];
      }
      // Weighted damped least squares in the scaled coordinates.
      const M js = j * scale_.asDiagonal();
      const M gram = js * js.transpose() + 1e-4 * M::Identity(js.rows(), js.rows());
      V delta = scale_.asDiagonal() * (js.transpose() * gram.ldlt().solve(e));
      const double ratio = (delta.cwiseQuotient(scale_)).cwiseAbs().maxCoeff();
      if (ratio > 0.5) {delta *= 0.5 / ratio;}
      x += delta;
      x.tail(joints_) = x.tail(joints_).cwiseMax(lower_).cwiseMin(upper_);
      x.segment<2>(3) = x.segment<2>(3).cwiseMax(tilt_lower_).cwiseMin(tilt_upper_);
    }
    return false;
  }
  double node_cost(const V & x, StpikSnapshot & probe, bool & safe, double & prevention)
  {
    const auto s = state(x);
    safe = true;
    prevention = 0.0;
    probe.backend->update(s);
    for (const auto & [index, hard] : considered_) {
      auto & task = *probe.tasks[static_cast<size_t>(index)];
      if (!task.enabled()) {continue;}
      task.update(s, *probe.backend);
      for (const auto & margin : StpikTaskAccess::preventive_margins(task, s)) {
        prevention += std::pow(std::max(0.0, margin.deficit), 2);
      }
      StpikAction::Result safety;
      if (!StpikTaskAccess::safety(task, s, simulation_.numerical_tolerance, safety)) {safe = false; return 0.0;}
    }
    const double clearance = scene_.clearance(s);
    if (clearance <= 2 * kMinimumClearance) {safe = false; return 0.0;}
    double cost = posture_.tilt * x.segment<2>(3).squaredNorm() / (kTiltScale * kTiltScale);
    for (int j = 0; j < joints_; ++j) {
      const double low = x(6 + j) - lower_(j), high = upper_(j) - x(6 + j), buffer = posture_.joint_buffer;
      if (low < buffer) {cost += posture_.joints * std::pow((buffer - low) / buffer, 2);}
      if (high < buffer) {cost += posture_.joints * std::pow((buffer - high) / buffer, 2);}
    }
    if (clearance < posture_.collision_buffer) {
      cost += posture_.clearance * std::pow((posture_.collision_buffer - clearance) / posture_.collision_buffer, 2);
    }
    return cost;
  }
  std::vector<V> layered(const std::vector<V> & seed, std::string & failure)
  {
    const int layers = std::clamp(static_cast<int>(options_.global_layers), 2, static_cast<int>(times_.size()) - 1);
    const int samples = std::max(4, static_cast<int>(options_.global_samples));
    std::mt19937 random(static_cast<unsigned>(options_.global_seed));
    std::uniform_real_distribution<double> unit(-1.0, 1.0);
    std::vector<double> tau;
    for (int l = 0; l <= layers; ++l) {tau.push_back(total_ * l / layers);}
    struct Node {V x; double cost; int parent; double prevention = 0.0;};
    const auto better = [](double ap, double ac, double bp, double bc) {
        return ap < bp || (ap == bp && ac < bc);
      };
    std::vector<std::vector<Node>> graph(static_cast<size_t>(layers + 1));
    graph[0].push_back({current_, 0.0, -1});
    auto probe = clone();
    if (free_start_) {
      bool safe = false;
      double prevention;
      const double cost = node_cost(current_, probe, safe, prevention);
      if (safe) {
        graph[0].front().cost = cost * dt_ / total_;
        graph[0].front().prevention = prevention * dt_ / total_;
      }
      // Alternative starting postures on the self-motion manifold of the path start.
      for (int i = 0; i < samples && !expired(); ++i) {
        V x = current_;
        for (int c = 0; c < 3; ++c) {x(c) += 0.5 * max_move_(c) * unit(random);}
        x(5) += 0.5 * max_move_(5) * unit(random);
        for (int j = 0; j < joints_; ++j) {x(6 + j) += std::min(0.8, 0.5 * max_move_(6 + j)) * unit(random);}
        x.tail(joints_) = x.tail(joints_).cwiseMax(lower_).cwiseMin(upper_);
        if (!project(x, 0.0, probe)) {continue;}
        const V move = x - current_;
        if ((move.cwiseAbs().array() > max_move_.array()).any()) {continue;}
        bool safe = false;
        double prevention;
        const double cost = node_cost(x, probe, safe, prevention);
        if (safe) {graph[0].push_back({x, posture_.movement * move.cwiseProduct(move_scale_).squaredNorm() + cost * dt_ / total_, -1, prevention * dt_ / total_});}
      }
    }
    for (int l = 1; l <= layers && !expired(); ++l) {
      const double span = tau[static_cast<size_t>(l)] - tau[static_cast<size_t>(l - 1)];
      std::vector<V> references;
      const auto valid_seed = !seed.empty() && std::all_of(seed.begin(), seed.end(), [](const V & v) {return v.size() > 0;});
      if (valid_seed) {references.push_back(interpolate(times_, seed, tau[static_cast<size_t>(l)]));}
      // Partial rollout states remain useful references.
      if (!rollout_.empty()) {
        std::vector<double> t; std::vector<V> k;
        for (size_t i = 0; i < rollout_.size(); ++i) {if (rollout_[i].size()) {t.push_back(times_[i]); k.push_back(rollout_[i]);}}
        if (t.size() > 1 && tau[static_cast<size_t>(l)] <= t.back()) {references.push_back(interpolate(t, k, tau[static_cast<size_t>(l)]));}
      }
      std::vector<const Node *> previous;
      for (const auto & node : graph[static_cast<size_t>(l - 1)]) {previous.push_back(&node);}
      std::sort(previous.begin(), previous.end(), [&](const Node * a, const Node * b) {return better(a->prevention, a->cost, b->prevention, b->cost);});
      for (size_t i = 0; i < std::min<size_t>(4, previous.size()); ++i) {references.push_back(previous[i]->x);}
      std::vector<V> candidates;
      for (int i = 0; i < samples && !expired(); ++i) {
        V x = references[static_cast<size_t>(i) % references.size()];
        if (i >= static_cast<int>(references.size())) {
          // Random self-motion around the reference: base position/yaw and joints.
          for (int c = 0; c < 3; ++c) {x(c) += 0.25 * unit(random);}
          x(5) += 0.4 * unit(random);
          for (int j = 0; j < joints_; ++j) {x(6 + j) += 0.8 * unit(random);}
          x.tail(joints_) = x.tail(joints_).cwiseMax(lower_).cwiseMin(upper_);
        }
        if (project(x, tau[static_cast<size_t>(l)], probe)) {candidates.push_back(x);}
      }
      for (const auto & x : candidates) {
        bool safe = false;
        double prevention;
        const double cost = node_cost(x, probe, safe, prevention);
        if (!safe) {continue;}
        double best_prevention = std::numeric_limits<double>::infinity();
        double best = std::numeric_limits<double>::infinity();
        int parent = -1;
        for (size_t p = 0; p < graph[static_cast<size_t>(l - 1)].size(); ++p) {
          const auto & from = graph[static_cast<size_t>(l - 1)][p];
          const V difference = x - from.x;
          if ((difference.cwiseAbs().array() > velocity_.array() * span).any()) {continue;}
          const double edge = difference.cwiseQuotient(scale_).squaredNorm() / span + cost * span / total_;
          const double preventive = from.prevention + prevention * span / total_;
          if (better(preventive, from.cost + edge, best_prevention, best)) {
            best_prevention = preventive; best = from.cost + edge; parent = static_cast<int>(p);
          }
        }
        if (parent >= 0) {graph[static_cast<size_t>(l)].push_back({x, best, parent, best_prevention});}
      }
      if (graph[static_cast<size_t>(l)].empty()) {
        std::ostringstream detail;
        detail << "layered search found no reachable safe configuration at t=" << tau[static_cast<size_t>(l)] << " s";
        if (!rollout_failure_.empty()) {detail << "; rollout={" << rollout_failure_ << "}";}
        failure = detail.str();
        return {};
      }
    }
    if (graph.back().empty()) {failure = "layered search interrupted"; return {};}
    const auto best = std::min_element(graph.back().begin(), graph.back().end(),
      [&](const Node & a, const Node & b) {return better(a.prevention, a.cost, b.prevention, b.cost);});
    std::vector<V> chain(static_cast<size_t>(layers + 1));
    int index = static_cast<int>(best - graph.back().begin());
    for (int l = layers; l >= 0; --l) {
      const auto & node = graph[static_cast<size_t>(l)][static_cast<size_t>(index)];
      chain[static_cast<size_t>(l)] = node.x; index = node.parent;
    }
    std::vector<V> out;
    for (double t : times_) {out.push_back(interpolate(tau, chain, t));}
    if (!free_start_) {out.front() = current_;}
    return out;
  }

  // --------------------------------------------------------------------------
  std::optional<NominalPlan> finish(const std::vector<V> & x, const std::string & method, std::string & failure)
  {
    if (expired()) {failure += failure.empty() ? "budget exhausted" : "; budget exhausted"; return std::nullopt;}
    NominalPlan plan;
    plan.times = times_;
    plan.knots = x;
    plan.method = method;
    std::vector<sura_manipulation_actions::BaseCoordinates> base;
    std::vector<V> joints;
    for (const auto & knot : x) {base.push_back(knot.head<6>()); joints.push_back(knot.tail(joints_));}
    std::string error;
    if (!sura_manipulation_actions::buildWholeBodyNominal(times_, base, joints, start_.model->all_joint_names(),
      start_.model->world_frame(), plan.joints, plan.base, error))
    {failure = error; return std::nullopt;}
    const auto evaluation = evaluate(x, false);
    plan.cost = evaluation.cost;
    plan.preventive_cost = evaluation.prevention;
    StpikSnapshot sim;
    if (free_start_ && !x.front().isApprox(current_, 1e-10)) {
      PosturePlan posture;
      posture.snapshot = start_;
      posture.request = goal_;
      posture.target = state(x.front());
      posture.options = posture_;
      posture.options.equality_tolerance = options_.equality_tolerance;
      posture.simulation = simulation_;
      posture.move = true;
      report("Nominal: STPIK repositioning check (" + method + ")");
      try {
        check_repositioning(posture, goal_, [this] {return expired();});
        sim = posture_snapshot(posture);
      } catch (const std::exception & error) {
        failure = (failure.empty() ? "" : failure + "; ") + method + " start posture rejected: " + error.what();
        start_rejected_ = true;
        report("Nominal: start posture rejected (" + method + "): " + error.what());
        return std::nullopt;
      }
    } else {
      sim = clone();
    }
    report("Nominal: STPIK validation (" + method + ")");
    auto validate_tasks = validate_posture_hierarchy(
      sim.tasks, goal_.task_id, simulation_.numerical_tolerance, options_.equality_tolerance);
    plan.validation = StpikCheck::run(std::move(sim), nominal_tracking_goal(goal_, plan), simulation_,
      [this] {return expired();}, {}, [&](const WholeBodyState & s, double) {
        if (!(scene_.clearance(s) > 0.0)) {throw std::runtime_error(scene_.collision_diagnostic());}
      }, {}, validate_tasks);
    if (plan.validation.outcome != plan.validation.SUCCESS) {
      std::ostringstream detail;
      detail << method << " rejected by STPIK: " << plan.validation.code << " at t="
             << plan.validation.time_from_start << " s: " << plan.validation.message;
      if (!plan.validation.task_id.empty()) {detail << "; task=" << plan.validation.task_id;}
      failure = failure.empty() ? detail.str() : failure + "; " + detail.str();
      return std::nullopt;
    }
    return plan;
  }

  const StpikSnapshot & start_;
  const Preparation::Goal & goal_;
  const StpikOptions & simulation_;
  const PostureOptions & posture_;
  const NominalOptions & options_;
  StpikCheck::Canceled canceled_;
  NominalFeedback feedback_;
  PostureScene scene_;
  std::chrono::steady_clock::time_point deadline_;
  sura_manipulation_actions::CartesianPosePath path_;
  bool free_start_ = false;
  bool start_rejected_ = false;
  int joints_ = 0, n_ = 0, selected_ = -1, nominal_ = -1;
  V current_, move_scale_, max_move_;
  V lower_, upper_, velocity_, scale_;
  Eigen::Vector2d tilt_lower_, tilt_upper_;
  Vector6 band_;
  std::vector<double> times_;
  double dt_ = 0.0, total_ = 0.0, last_cost_ = 0.0;
  std::vector<std::pair<int, bool>> considered_;  // task index, hard
  std::vector<V> rollout_;
  std::string rollout_failure_;
};
}  // namespace

bool NominalOptions::valid(std::string & reason) const
{
  if (!std::isfinite(equality_tolerance) || equality_tolerance <= 0.0) {
    reason = "Nominal equality tolerance must be finite and positive";
    return false;
  }
  for (double positive : {budget, knot_dt, trust_position, trust_angle, trust_joint, velocity_fraction}) {
    if (!std::isfinite(positive) || positive <= 0.0) {reason = "Nominal budget, knot step, trust region and velocity fraction must be positive"; return false;}
  }
  for (double nonnegative : {smooth_weight, soft_task_weight}) {
    if (!std::isfinite(nonnegative) || nonnegative < 0.0) {reason = "Nominal weights must be finite and nonnegative"; return false;}
  }
  if (velocity_fraction > 1.0) {reason = "Nominal velocity fraction cannot exceed 1"; return false;}
  if (!(max_knots >= 2) || !(iterations >= 1) || !(global_layers >= 2) || !(global_samples >= 4) ||
    !std::isfinite(global_seed) || global_seed < 0)
  {reason = "Nominal knot, iteration and sampling counts are too small"; return false;}
  return true;
}

StpikAction::Goal nominal_tracking_goal(const Preparation::Goal & goal, const NominalPlan & plan)
{
  auto out = posture_tracking_goal(goal);
  out.nominal_joints = plan.joints;
  out.nominal_base = plan.base;
  return out;
}

NominalPlan optimize_nominal_trajectory(
  const StpikSnapshot & start, const Preparation::Goal & goal, const StpikOptions & simulation,
  const PostureOptions & posture, const NominalOptions & options,
  const StpikCheck::Canceled & canceled, const NominalFeedback & feedback, const NominalPlan * warm,
  bool free_start)
{
  std::string reason;
  if (!options.valid(reason)) {throw std::runtime_error(reason);}
  Planner planner(start, goal, simulation, posture, options, canceled, feedback, free_start && !warm);
  return planner.plan(warm);
}
}  // namespace task_priority_kinematic_control
