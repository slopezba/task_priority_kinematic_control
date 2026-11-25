#include "task_priority_kinematic_control/core/initial_posture.hpp"
#include "task_priority_kinematic_control/core/posture_scene.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "task_priority_kinematic_control/tasks/joint_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/robot_nominal_task.hpp"
#include "sura_manipulation_actions/dense_qp.hpp"
#include <cmath>
#include <limits>
#include <algorithm>
#include <sstream>

namespace task_priority_kinematic_control
{
namespace
{
using V = Eigen::VectorXd;
using M = Eigen::MatrixXd;
using Pose = geometry_msgs::msg::Pose;
constexpr double kUnboundedStep = 1e10;
Pose message(const Eigen::Isometry3d & t)
{
  Pose p;
  p.position.x = t.translation().x();
  p.position.y = t.translation().y();
  p.position.z = t.translation().z();
  Eigen::Quaterniond q(t.linear());
  p.orientation.x = q.x();
  p.orientation.y = q.y();
  p.orientation.z = q.z();
  p.orientation.w = q.w();
  return p;
}
Eigen::Isometry3d transform(const Pose & p)
{
  Eigen::Isometry3d t = Eigen::Isometry3d::Identity();
  const auto & q = p.orientation;
  Eigen::Quaterniond r(q.w, q.x, q.y, q.z);
  t.translation() << p.position.x, p.position.y, p.position.z;
  if (!t.translation().allFinite() || std::abs(r.norm() - 1) > 1e-6 || !r.coeffs().allFinite())
  {throw std::runtime_error("Invalid posture pose");}
  t.linear() = r.normalized().toRotationMatrix();
  return t;
}
std::shared_ptr<TaskBase> find(const TaskList & tasks, const std::string & id)
{
  for (const auto & t:tasks) {
    if (t->id() == id) {
      return t;
    }
  }
  throw std::runtime_error("Missing posture task: " + id);
}
StpikSnapshot clone(const StpikSnapshot & s)
{
  auto c = StpikTaskAccess::capture(s.initial_state, s.model, s.backend, s.solver, s.tasks);
  if (!c.unavailable_reason.empty()) {throw std::runtime_error(c.unavailable_reason);}
  c.captured = s.captured;
  return c;
}
using Scene = PostureScene;
V coords(const WholeBodyState & s)
{
  V v(6 + s.joint_positions.size());
  v << s.base_position, s.base_rpy, s.joint_positions;
  return v;
}
WholeBodyState with_coordinates(WholeBodyState s, const V & x)
{
  s.base_position = x.head<3>();
  s.base_rpy = x.segment<3>(3);
  s.joint_positions = x.tail(s.joint_positions.size());
  s.joint_velocities.setZero();
  return s;
}
StpikAction::Goal tracking(const Preparation::Goal & g)
{
  StpikAction::Goal c;
  c.trajectory = g.trajectory;
  c.task_id = g.task_id;
  c.tip_frame = g.tip_frame;
  c.tolerance_orientation = g.tolerance_orientation;
  c.path_tolerance = g.path_tolerance;
  c.goal_tolerance = g.goal_tolerance;
  c.goal_time_tolerance = g.goal_time_tolerance;
  return c;
}
}
bool tip_within(const Eigen::Isometry3d & measured, const Preparation::Goal & g)
{
  auto expected = transform(g.trajectory.start_pose);
  const auto & q = g.tolerance_orientation;
  Eigen::Matrix3d axes = Eigen::Quaterniond(q.w, q.x, q.y, q.z).toRotationMatrix().transpose();
  Eigen::AngleAxisd angle(expected.linear() * measured.linear().transpose());
  Eigen::Matrix<double, 6, 1> e;
  e << axes * (expected.translation() - measured.translation()),
      axes * (angle.axis() * angle.angle());
  for (int i = 0; i < 6; ++i) {
    if (!std::isfinite(e[i]) || std::abs(e[i]) > g.path_tolerance[i] + 1e-6) {
      return false;
    }
  }
  return true;
}
bool posture_reached(
  const WholeBodyState & a, const WholeBodyState & b, const PostureOptions & options)
{
  const Eigen::Matrix3d rotation_error = pose_math::base_pose_from_state(a).linear() *
    pose_math::base_pose_from_state(b).linear().transpose();
  return (a.base_position - b.base_position).norm() < options.position_tolerance &&
         Eigen::AngleAxisd(rotation_error).angle() < options.orientation_tolerance &&
         (a.joint_positions - b.joint_positions).lpNorm<Eigen::Infinity>() < options.joint_tolerance;
}
namespace
{
bool legacy_posture_hold(const std::string & id)
{
  return id == "base_pose" || id == "left_arm_nominal" || id == "right_arm_nominal";
}
}

TaskList posture_command_tasks(const TaskList & tasks, const std::string & selected)
{
  // With a RobotNominalTask below the tool task the posture is streamed to it, and its
  // sync_tasks holds follow every reference. Otherwise the legacy holds are driven.
  TaskList below, legacy;
  std::shared_ptr<RobotNominalTask> nominal;
  bool below_selected = false;
  for (const auto & task : tasks) {
    if (task->id() == selected) {
      below_selected = true;
    } else if (below_selected) {
      below.push_back(task);
      if (!nominal) {nominal = std::dynamic_pointer_cast<RobotNominalTask>(task);}
      if (legacy_posture_hold(task->id())) {legacy.push_back(task);}
    }
  }
  if (!nominal) {return legacy;}
  TaskList commands{nominal};
  for (const auto & task : below) {
    const auto & sync = nominal->sync_tasks();
    if (std::find(sync.begin(), sync.end(), task->id()) != sync.end()) {commands.push_back(task);}
  }
  return commands;
}

TaskList prepare_tasks(const TaskList & tasks, const WholeBodyModel & m, const Preparation::Goal & g)
{
  if (g.tip_frame != m.left_tip_frame() && g.tip_frame != m.right_tip_frame()) {
    throw std::runtime_error("Unknown tool frame");
  }
  const auto active = std::dynamic_pointer_cast<PoseTaskCommon>(find(tasks, g.task_id));
  if (!active || !active->enabled() || !find(tasks, "stabilize")->enabled()) {
    throw std::runtime_error("Required pose/level tasks unavailable");
  }
  for (const auto & id : {"joint_limits", "self_collision_avoidance", "stabilize_limits"}) {
    if (!find(tasks, id)->enabled()) {
      throw std::runtime_error(std::string("Required safety task disabled: ") + id);
    }
  }
  const auto commands = posture_command_tasks(tasks, g.task_id);
  const auto hold = transform(g.trajectory.start_pose);
  if (!commands.empty()) {
    if (const auto nominal = std::dynamic_pointer_cast<RobotNominalTask>(commands.front())) {
      // Enabled holds that robot_nominal does not retarget would fight the posture.
      bool below_selected = false;
      for (const auto & task : tasks) {
        if (task->id() == g.task_id) {below_selected = true; continue;}
        if (!below_selected || !task->enabled() || !legacy_posture_hold(task->id())) {continue;}
        if (std::find(commands.begin(), commands.end(), task) == commands.end()) {
          throw std::runtime_error("Posture over robot_nominal: enabled hold " + task->id() +
            " must be listed in tasks.robot_nominal.sync_tasks");
        }
      }
      // Drive the whole body: base x, y, z, yaw and every joint (roll/pitch stay with stabilize).
      std::vector<bool> activation(nominal->rows(), true);
      activation[3] = activation[4] = false;
      std::string error;
      active->capture_measured_target(hold);
      nominal->set_enabled(true);
      if (!nominal->set_joint_activation(activation, error)) {throw std::runtime_error(error);}
      return commands;
    }
  }
  for (const auto & task : commands) {
    if (task->id() == "base_pose") {
      if (!std::dynamic_pointer_cast<FramePoseTask>(task)) {
        throw std::runtime_error("Expected a FramePoseTask for base_pose");
      }
    } else {
      const auto & expected =
        task->id() == "left_arm_nominal" ? m.left_arm_joints() : m.right_arm_joints();
      if (!std::dynamic_pointer_cast<JointNominalTask>(task) ||
        task->build_status().joint_names != expected)
      {
        throw std::runtime_error("Invalid posture command task: " + task->id());
      }
    }
  }
  active->capture_measured_target(hold);
  for (const auto & task : commands) {
    task->set_enabled(true);
    if (auto arm = std::dynamic_pointer_cast<JointNominalTask>(task)) {
      arm->clear_velocity();
      std::string error;
      const auto count = arm->build_status().joint_names.size();
      if (!arm->set_joint_activation(std::vector<bool>(count, true), error)) {
        throw std::runtime_error(error);
      }
    }
  }
  return commands;
}

void posture_reference(
  const TaskList & commands, const WholeBodyModel & m, const WholeBodyState & start,
  const WholeBodyState & target, double f, double duration)
{
  f = std::clamp(f, 0., 1.);
  // Quintic profile s(f) and its time derivative ds/dt = s'(f) / duration.
  const double rate = duration > 0. ? 30. * f * f * (1. - f) * (1. - f) / duration : 0.;
  const V delta = coords(target) - coords(start);
  f = f * f * f * (10. + f * (-15. + 6. * f));
  const auto ref = with_coordinates(start, coords(start) + f * delta);
  if (!commands.empty()) {
    if (const auto nominal = std::dynamic_pointer_cast<RobotNominalTask>(commands.front())) {
      const V x = coords(ref), velocity = rate * delta;
      const auto joints = static_cast<Eigen::Index>(ref.joint_positions.size());
      std::string error;
      // The commands list holds the sync_tasks, which are retargeted with the reference.
      if (!apply_robot_nominal_reference(commands, *nominal, x.head<6>(), velocity.head<6>(),
          x.tail(joints), velocity.tail(joints), error))
      {
        throw std::runtime_error("robot_nominal rejected the posture reference: " + error);
      }
      return;
    }
  }
  for (const auto & task : commands) {
    if (auto base = std::dynamic_pointer_cast<FramePoseTask>(task)) {
      base->capture_measured_target(pose_math::base_pose_from_state(ref));
    } else {
      const auto & names =
        task->id() == "left_arm_nominal" ? m.left_arm_joints() : m.right_arm_joints();
      std::vector<double> values;
      std::string error;
      for (const auto & name : names) {
        values.push_back(ref.joint_positions(m.joint_index(name)));
      }
      if (!task->set_joint_target(values, error)) {throw std::runtime_error(error);}
    }
  }
}

void restore_posture_commands(const TaskList & commands, const TaskList & original)
{
  for (const auto & task : commands) {
    const auto source = find(original, task->id());
    task->set_enabled(source->enabled());
    if (std::dynamic_pointer_cast<JointNominalTask>(task) ||
      std::dynamic_pointer_cast<RobotNominalTask>(task))
    {
      std::string error;
      if (!task->set_joint_activation(source->build_status().joint_activation, error)) {
        throw std::runtime_error(error);
      }
    }
  }
}

namespace
{
void check_region(
  const TaskBase & task, const PostureRegion & region, double numerical_tolerance)
{
  for (Eigen::Index axis = 0; axis < region.limit.size(); ++axis) {
    const double measured = region.radial ? region.error.norm() : std::abs(region.error(axis));
    if (!std::isfinite(measured) || measured > region.limit(axis) + numerical_tolerance) {
      std::ostringstream detail;
      detail << "Higher-priority task " << task.id() << " exceeded "
             << (region.radial ? "norm limit" : "pose limit") << " on row " << axis
             << "; measured=" << measured << "; limit=" << region.limit(axis);
      throw std::runtime_error(detail.str());
    }
  }
}
}

StpikCheck::ValidateTasks validate_posture_hierarchy(
  const TaskList & tasks, const std::string & selected, double numerical_tolerance, double equality_tolerance)
{
  const auto end = std::find_if(tasks.begin(), tasks.end(), [&](const auto & task) {
    return task->id() == selected;
  });
  TaskList higher(tasks.begin(), end);
  return [higher = std::move(higher), numerical_tolerance, equality_tolerance, initial = std::vector<V>{}](
      const std::vector<TaskComputation> & computations) mutable {
      const bool first = initial.empty();
      if (first) {initial.resize(higher.size());}
      for (size_t i = 0; i < higher.size(); ++i) {
        const auto & task = *higher[i];
        if (!task.enabled()) {continue;}
        PostureRegion region;
        if (StpikTaskAccess::posture_region(task, region)) {
          if (region.always_active) {
            if (first) {initial[i] = region.error;}
            region.limit(0) = std::max(equality_tolerance, initial[i].norm());
          }
          check_region(task, region, numerical_tolerance);
          continue;
        }
        const auto & computation = computations.at(i);
        if (!computation.active || computation.structural_zero_rows) {continue;}
        const V error = computation.error.cwiseAbs();
        if (first) {initial[i] = error;}
        for (Eigen::Index row = 0; row < error.size(); ++row) {
          // Allow the existing tracking error to settle without requiring a perfect
          // starting measurement or allowing the nominal motion to worsen it.
          const double start_error = row < initial[i].size() ? initial[i](row) : 0.0;
          const double limit = std::max(equality_tolerance, start_error) + numerical_tolerance;
          if (!std::isfinite(error(row)) || error(row) > limit) {
            std::ostringstream detail;
            detail << "Higher-priority task " << task.id() << " exceeded tracking tolerance"
                   << " on row " << row << "; measured=" << error(row) << "; limit=" << limit;
            throw std::runtime_error(detail.str());
          }
        }
      }
    };
}

namespace
{
struct PostureLinearization
{
  struct Row
  {
    Eigen::RowVectorXd jacobian;
    double lower;
    double upper;
  };
  std::vector<Row> rows;
  std::vector<PostureRegion> radial;
};

std::vector<TaskComputation> update_higher_tasks(
  StpikSnapshot & snapshot, const WholeBodyState & state, const std::string & selected)
{
  snapshot.backend->update(state);
  std::vector<TaskComputation> computations;
  for (const auto & task : snapshot.tasks) {
    if (task->id() == selected) {break;}
    computations.push_back(task->update(state, *snapshot.backend));
  }
  return computations;
}

PostureLinearization linearize_hierarchy(
  StpikSnapshot & snapshot, const WholeBodyState & state, const std::string & selected,
  const Eigen::Matrix3d & spatial)
{
  const auto computations = update_higher_tasks(snapshot, state, selected);
  PostureLinearization out;
  for (size_t i = 0; i < computations.size(); ++i) {
    const auto & task = *snapshot.tasks[i];
    if (!task.enabled()) {continue;}
    PostureRegion region;
    if (StpikTaskAccess::posture_region(task, region, true)) {
      region.jacobian.middleCols<3>(3) *= spatial;
      for (Eigen::Index axis = 0; axis < region.error.size(); ++axis) {
        const double limit = region.always_active ? 0.0 : region.limit(region.radial ? 0 : axis);
        out.rows.push_back({region.jacobian.row(axis),
          region.error(axis) - limit, region.error(axis) + limit});
      }
      if (region.radial && !region.always_active) {out.radial.push_back(std::move(region));}
      continue;
    }
    const auto & computation = computations[i];
    if (!computation.active) {continue;}
    for (Eigen::Index row = 0; row < computation.jacobian.rows(); ++row) {
      Eigen::RowVectorXd jacobian = computation.jacobian.row(row);
      if (jacobian.isZero(1e-12)) {continue;}
      jacobian.segment<3>(3) *= spatial;
      const double error = computation.error(row);
      // Reserve tracking rows, and allow set-based safety rows to move inward
      // without requiring every active limit to settle at its hysteresis target.
      const double lower = computation.structural_zero_rows ?
        (error > 0.0 ? 0.0 : -kUnboundedStep) : error;
      const double upper = computation.structural_zero_rows ?
        (error < 0.0 ? 0.0 : kUnboundedStep) : error;
      out.rows.push_back({std::move(jacobian), lower, upper});
    }
  }
  return out;
}

bool solve_posture_step(
  const M & identity, const V & gradient, const M & constraints, const V & lower, const V & upper,
  const PostureLinearization & hierarchy, V & step)
{
  auto rows = hierarchy.rows;
  const auto fixed_rows = constraints.rows();
  // OSQP takes linear constraints. Add a supporting plane at each predicted
  // violation of the local norm ball, including when the current error is zero.
  for (int iteration = 0; iteration < 8; ++iteration) {
    M a(fixed_rows + rows.size(), constraints.cols());
    V lb(a.rows()), ub(a.rows());
    a.topRows(fixed_rows) = constraints;
    lb.head(fixed_rows) = lower;
    ub.head(fixed_rows) = upper;
    for (size_t i = 0; i < rows.size(); ++i) {
      a.row(fixed_rows + i) = rows[i].jacobian;
      lb(fixed_rows + i) = rows[i].lower;
      ub(fixed_rows + i) = rows[i].upper;
    }
    if (!sura_manipulation_actions::qp::solveQp(identity, gradient, a, lb, ub, 0.05, step)) {
      return false;
    }
    const auto count = rows.size();
    for (const auto & region : hierarchy.radial) {
      const V error = region.error - region.jacobian * step;
      const double radius = region.limit(0);
      if (error.norm() <= radius + 1e-6) {continue;}
      const V direction = error.normalized();
      rows.push_back({direction.transpose() * region.jacobian,
        direction.dot(region.error) - radius, kUnboundedStep});
    }
    if (rows.size() == count) {return true;}
  }
  return false;
}

std::string simulation_failure(const std::string & phase, const StpikAction::Result & report)
{
  std::ostringstream detail;
  detail << phase << ": " << report.code << " at t=" << report.time_from_start
         << " s: " << report.message;
  if (!report.task_id.empty()) {detail << "; task=" << report.task_id;}
  if (!report.constraint.empty()) {detail << "; constraint=" << report.constraint;}
  detail << "; tip_error=[";
  for (size_t i = 0; i < report.tracking_error.size(); ++i) {
    if (i) {detail << ", ";}
    detail << report.tracking_error[i];
  }
  detail << "]";
  return detail.str();
}
PostureSummary evaluate(
  const PosturePlan & p, const WholeBodyState & target, Scene & scene,
  const V & lower, const V & upper, const StpikCheck::Canceled & canceled,
  bool score_path = true)
{
  auto check = tracking(p.request);
  auto start = p.snapshot.initial_state;
  const bool move = !coords(start).isApprox(coords(target), 1e-10);
  auto sim = clone(p.snapshot);
  auto validate_tasks = validate_posture_hierarchy(
    sim.tasks, p.request.task_id, p.simulation.numerical_tolerance, p.options.equality_tolerance);
  PostureSafetyCosts safety_costs;
  double obstacle_cost = 0.0;
  bool score_trajectory = false;
  // Keep the cloned tasks alive after the snapshot is consumed by StpikCheck.
  const auto inspect_safety = [&, tasks = sim.tasks](const WholeBodyState & state) {
      for (const auto & task : tasks) {
        const auto costs = StpikTaskAccess::posture_safety_costs(*task, state, p.options.active_penalty);
        safety_costs.joints = std::max(safety_costs.joints, costs.joints);
        safety_costs.stabilize = std::max(safety_costs.stabilize, costs.stabilize);
        safety_costs.self_collision = std::max(safety_costs.self_collision, costs.self_collision);
      }
    };
  auto inspect = [&](const WholeBodyState & s) {
      const double d = scene.clearance(s);
      if (!(d > 0.)) {throw std::runtime_error(scene.collision_diagnostic());}
      if (!score_trajectory) {return d;}
      inspect_safety(s);
      const double world_distance = scene.world_clearance();
      obstacle_cost = std::max(obstacle_cost, posture_limit_cost(
        world_distance, p.options.obstacle_activation_distance, p.options.collision_buffer,
        world_distance <= p.options.obstacle_activation_distance, p.options.active_penalty));
      return d;
    };
  if (move) {
    const auto commands = prepare_tasks(sim.tasks, *sim.model, p.request);
    if (commands.empty()) {
      throw std::runtime_error("No posture command tasks below " + p.request.task_id);
    }
    auto hold = check;
    hold.trajectory.target_pose = hold.trajectory.start_pose;
    hold.trajectory.duration = p.options.duration + p.options.settle;
    hold.goal_tolerance = hold.path_tolerance;
    hold.goal_time_tolerance = 0;
    const auto r = StpikCheck::run(
      sim, hold, p.simulation, canceled, {},
      [&](const auto & s, double) {inspect(s); start = s;},
      [&](const auto &, double t) {
        posture_reference(commands, *sim.model, p.snapshot.initial_state, target, t / p.options.duration,
          p.options.duration);
      }, validate_tasks);
    if (r.outcome != r.SUCCESS) {
      throw std::runtime_error(simulation_failure("Repositioning failed", r));
    }
    if (!posture_reached(start, target, p.options)) {
      std::ostringstream detail;
      detail << "Repositioning settling tolerance exceeded after "
             << p.options.duration + p.options.settle << " s"
             << "; base_position_error=" << (start.base_position-target.base_position).norm()
             << " m; maximum_joint_error="
             << (start.joint_positions-target.joint_positions).lpNorm<Eigen::Infinity>() << " rad";
      throw std::runtime_error(detail.str());
    }
    // Check the exact candidate geometry as well as the states reached by
    // simulation. A linearized region or a settling tolerance can hide a
    // small violation in the nominal reference itself.
    auto exact = clone(sim);
    const auto higher = update_higher_tasks(exact, target, p.request.task_id);
    for (size_t i = 0; i < higher.size(); ++i) {
      const auto & task = *exact.tasks[i];
      PostureRegion region;
      if (task.enabled() && StpikTaskAccess::posture_region(task, region)) {
        if (region.always_active) {region.limit.setConstant(p.options.equality_tolerance);}
        try {
          check_region(task, region, p.simulation.numerical_tolerance);
        } catch (const std::runtime_error & error) {
          throw std::runtime_error("Nominal posture rejected: " + std::string(error.what()));
        }
      }
    }
    restore_posture_commands(commands, p.snapshot.tasks);
    sim.captured += std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(hold.trajectory.duration));
  }
  // Repositioning-only checks (whole-trajectory nominal mode validates the path itself).
  if (!score_path) {return PostureSummary();}
  sim.initial_state = start;
  score_trajectory = true;
  PostureSummary out;
  out.minimum_clearance = out.minimum_joint_margin = std::numeric_limits<double>::infinity();
  double integral = 0., previous_time = 0., previous_tilt = 0.;
  const auto report = StpikCheck::run(
    std::move(sim), check, p.simulation, canceled, {},
    [&](const WholeBodyState & s, double time) {
      out.minimum_clearance = std::min(out.minimum_clearance, inspect(s));
      const double margin =
      std::min((s.joint_positions - lower).minCoeff(), (upper - s.joint_positions).minCoeff());
      out.minimum_joint_margin = std::min(out.minimum_joint_margin, margin);
      const double squared = s.base_rpy.head<2>().squaredNorm();
      integral += 0.5 * (squared + previous_tilt) * (time - previous_time);
      previous_time = time;
      previous_tilt = squared;
      out.peak_tilt = std::max(out.peak_tilt, std::sqrt(squared));
    }, {}, validate_tasks);
  if (report.outcome != report.SUCCESS) {
    throw std::runtime_error(simulation_failure("Trajectory validation failed", report));
  }
  out.minimum_clearance = std::min(out.minimum_clearance, report.minimum_clearance);
  // Use physical scales for translation, orientation, and each joint's travel.
  V delta = coords(target);
  delta -= coords(p.snapshot.initial_state);
  delta.head<3>() /= p.options.radius;
  delta.tail(lower.size()).array() /= (upper - lower).array();
  out.cost = p.options.movement * delta.squaredNorm() +
    p.options.tilt * integral / std::max(previous_time, p.simulation.dt) +
    p.options.joints * safety_costs.joints +
    p.options.stabilize * safety_costs.stabilize +
    p.options.self_collision * safety_costs.self_collision +
    p.options.clearance * obstacle_cost;
  out.configuration.base.header.frame_id = p.snapshot.model->world_frame();
  out.configuration.base.pose = message(pose_math::base_pose_from_state(target));
  out.configuration.joint_names = p.snapshot.model->all_joint_names();
  out.configuration.positions.assign(
    target.joint_positions.data(),
    target.joint_positions.data() + lower.size());
  return out;
}
}
PosturePlan optimize_posture(
  StpikSnapshot s, const Preparation::Goal & goal, const StpikOptions & options,
  const PostureOptions & limits, const StpikCheck::Canceled & canceled,
  const std::function<void(unsigned)> & feedback)
{
  if (!s.unavailable_reason.empty() || !s.model || !s.backend) {
    throw std::runtime_error("Snapshot unavailable: " + s.unavailable_reason);
  }
  V lo, hi;
  if (!StpikTaskAccess::joint_bounds(s.tasks, lo, hi) || lo.size() != s.initial_state.joint_positions.size() ||
    lo.size() == 0 || s.model->active_base_dofs() != std::vector<int>({0, 1, 2, 3, 4, 5}))
  {
    throw std::runtime_error(
            "Full floating base and both arms with enabled finite joint limits are required");
  }
  if (!s.initial_state.navigation_valid || !s.initial_state.joints_valid ||
    !coords(s.initial_state).allFinite() || !s.initial_state.joint_velocities.allFinite())
  {
    throw std::runtime_error("Posture search requires a complete finite controller state");
  }
  auto preflight = clone(s);
  const auto commands = prepare_tasks(preflight.tasks, *s.model, goal);
  // Initialize any unset higher-priority goals at the measured starting state.
  update_higher_tasks(preflight, s.initial_state, goal.task_id);
  Scene scene(goal, *s.model);
  PosturePlan p;
  p.snapshot = std::move(s);
  p.request = goal;
  p.simulation = options;
  p.options = limits;
  const auto end = std::chrono::steady_clock::now() + std::chrono::duration<double>(limits.budget);
  auto stop = [&] {return canceled() || std::chrono::steady_clock::now() >= end;};
  unsigned calls = 0;
  std::string last, baseline_failure;
  unsigned completed = 0;
  bool feasible = false, baseline = false;
  double initial_cost = std::numeric_limits<double>::infinity();
  auto attempt = [&](const WholeBodyState & candidate) {
      feedback(++calls);
      try {
        auto report = evaluate(p, candidate, scene, lo, hi, stop);
        ++completed;
        if (!feasible || report.cost + 1e-6 < p.summary.cost) {
          feasible = true;
          p.summary = report;
          p.target = candidate;
          return true;
        }
      } catch (const std::exception & e) {
        // The deadline interrupts an unfinished evaluation. Keep the last completed
        // rejection so a time limit cannot hide its tracking or safety diagnostic.
        if (!stop()) {
          ++completed;
          last = e.what();
        }
      }
      return false;
    };
  const auto & start = p.snapshot.initial_state;
  p.target = start;
  if (attempt(start)) {baseline = true; initial_cost = p.summary.cost;}
  else {baseline_failure = last.empty() ? "evaluation interrupted before completion" : last;}
  const V origin = coords(start);
  const int n = origin.size();
  const M identity = M::Identity(n, n);
  for (double radius = limits.radius;
    !commands.empty() && radius >= limits.radius / 8 && !stop() &&
    !(feasible && p.summary.cost <= 1e-12);
    radius *= 0.5)
  {
    auto center = p.target;
    const V x = coords(center);
    const auto frame = pose_math::compute_tip_state(*p.snapshot.backend, center, goal.tip_frame,
      find(p.snapshot.tasks, goal.task_id).get());
    M a = M::Zero(n + 7, n);
    a.topRows(n) = identity;
    a.middleRows(n, 6) = frame.jacobian;
    // Convert Euler coordinate increments to the spatial angular displacement used by KDL.
    Eigen::Matrix3d spatial;
    const Eigen::AngleAxisd yaw(center.base_rpy.z(), Eigen::Vector3d::UnitZ());
    const Eigen::AngleAxisd pitch(center.base_rpy.y(), Eigen::Vector3d::UnitY());
    spatial.col(0) = yaw * pitch * Eigen::Vector3d::UnitX();
    spatial.col(1) = yaw * Eigen::Vector3d::UnitY();
    spatial.col(2) = Eigen::Vector3d::UnitZ();
    a.block(n, 3, 6, 3) = frame.jacobian.middleCols(3, 3) * spatial;
    auto projected_tasks = clone(preflight);
    const auto hierarchy = linearize_hierarchy(projected_tasks, center, goal.task_id, spatial);
    const auto expected = transform(goal.trajectory.start_pose);
    Eigen::Matrix<double, 6, 1> err;
    Eigen::AngleAxisd rotation_error(expected.linear() * frame.pose.linear().transpose());
    err << expected.translation() - frame.pose.translation(),
        rotation_error.axis() * rotation_error.angle();
    V lb = V::Constant(n + 7, -radius), ub = V::Constant(n + 7, radius);
    lb.segment(6, lo.size()) = lb.segment(6, lo.size()).cwiseMax(lo - center.joint_positions);
    ub.segment(6, lo.size()) = ub.segment(6, lo.size()).cwiseMin(hi - center.joint_positions);
    lb.segment<6>(n) = ub.segment<6>(n) = err;
    const double distance = scene.clearance(center);
    lb[n + 6] = -1e10;
    ub[n + 6] = 1e10;
    if (distance < limits.collision_buffer) {
      for (int j = 0; j < n; ++j) {
        V perturbed = x;
        perturbed[j] += 1e-4;
        a(n + 6, j) = (scene.clearance(with_coordinates(center, perturbed)) - distance) / 1e-4;
      }
      lb[n + 6] = 0.001 - distance;
    }
    const auto try_step = [&](const V & gradient) {
        V step;
        if (!solve_posture_step(identity, gradient, a, lb, ub, hierarchy, step) ||
            step.norm() < 1e-5) {return;}
        auto candidate = with_coordinates(center, x + step);
        const auto tip = pose_math::compute_tip_state(*p.snapshot.backend, candidate, goal.tip_frame,
          find(p.snapshot.tasks, goal.task_id).get());
        if (tip_within(tip.pose, goal)) {
          attempt(candidate);
        }
      };
    // Try increasing the most threatened margins first, before polling individual directions.
    V gradient = V::Zero(n);
    gradient.segment<2>(3) = limits.tilt * center.base_rpy.head<2>();
    for (Eigen::Index j = 0; j < lo.size(); ++j) {
      const double low = center.joint_positions[j] - lo[j],
        high = hi[j] - center.joint_positions[j];
      gradient[6 + j] = radius * limits.joints * (std::max(0., 1. - high / limits.joint_buffer) -
        std::max(0., 1. - low / limits.joint_buffer));
    }
    if (distance < limits.collision_buffer) {
      gradient -= radius * limits.clearance * a.row(n + 6).transpose();
    }
    if (!stop()) {try_step(gradient);}
    for (int axis = 0; axis < n && !stop(); ++axis) {
      for (double sign:{1., -1.}) {
        if (stop()) {break;}
        gradient.setZero();
        gradient[axis] = -sign * radius;
        try_step(gradient);
      }
    }
  }
  if (canceled()) {throw std::runtime_error("Posture search canceled");}
  if (!feasible) {
    std::ostringstream detail;
    detail << (std::chrono::steady_clock::now() >= end ?
      "Posture search time budget exhausted" : "Posture candidate search exhausted")
           << "; budget=" << limits.budget << " s; evaluations_started=" << calls
           << "; evaluations_completed=" << completed
           << "; initial_posture={" << baseline_failure << "}";
    if (!last.empty() && last != baseline_failure) {detail << "; last_rejection={" << last << "}";}
    throw std::runtime_error(detail.str());
  }
  p.summary.baseline_feasible = baseline;
  p.summary.baseline_cost = initial_cost;
  p.move = !coords(p.target).isApprox(origin, 1e-10);
  return p;
}
StpikAction::Goal posture_tracking_goal(const Preparation::Goal & g) {return tracking(g);}

StpikSnapshot posture_snapshot(const PosturePlan & p)
{
  auto sim = clone(p.snapshot);
  if (!coords(p.snapshot.initial_state).isApprox(coords(p.target), 1e-10)) {
    const auto commands = prepare_tasks(sim.tasks, *sim.model, p.request);
    posture_reference(commands, *sim.model, p.snapshot.initial_state, p.target, 1.0);
    restore_posture_commands(commands, p.snapshot.tasks);
  }
  sim.initial_state = with_coordinates(p.snapshot.initial_state, coords(p.target));
  return sim;
}

void check_repositioning(
  const PosturePlan & p, const Preparation::Goal & fresh, const StpikCheck::Canceled & canceled)
{
  Scene scene(fresh, *p.snapshot.model);
  V lower, upper;
  if (!StpikTaskAccess::joint_bounds(p.snapshot.tasks, lower, upper)) {
    throw std::runtime_error("Missing joint limits");
  }
  evaluate(p, p.target, scene, lower, upper, canceled, false);
}

void recheck_posture(
  const PosturePlan & p, const Preparation::Goal & fresh,
  const StpikCheck::Canceled & canceled)
{
  Scene scene(fresh, *p.snapshot.model);
  V lower, upper;
  if (!StpikTaskAccess::joint_bounds(p.snapshot.tasks, lower, upper)) {
    throw std::runtime_error("Missing joint limits");
  }
  evaluate(p, p.target, scene, lower, upper, canceled);
}
}
