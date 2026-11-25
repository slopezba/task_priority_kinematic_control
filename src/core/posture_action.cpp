#include "task_priority_kinematic_control/core/posture_action.hpp"
#include "task_priority_kinematic_control/core/stpik_check_io.hpp"
#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"
#include <cmath>
#include <limits>

namespace task_priority_kinematic_control
{
PostureAction::PostureAction(
  rclcpp_lifecycle::LifecycleNode::SharedPtr node, std::mutex & mutex,
  WholeBodyState & state, std::shared_ptr<WholeBodyModel> & model, KinematicsBackendPtr & backend,
  HierarchySolver & solver, TaskManager & manager, CartesianTrajectoryCoordinator & coordinator,
  std::function<bool()> ready)
: node_(node), mutex_(mutex), state_(state), model_(model), backend_(backend), solver_(solver),
  manager_(manager), coordinator_(coordinator), ready_(std::move(ready))
{
  for (const auto & pair:std::vector<std::pair<std::string, double>>{
      {"budget", 30}, {"radius", 0.15}, {"duration", 6}, {"settle", 4}, {"movement_weight", 0.05},
      {"tilt_weight", 1}, {"joint_weight", 1}, {"clearance_weight", 1}, {"joint_buffer", 0.20},
      {"collision_buffer", 0.06}, {"position_tolerance", 0.01},
      {"orientation_tolerance", 0.035}, {"joint_tolerance", 0.035},
      {"stabilize_weight", 1}, {"self_collision_weight", 1}, {"activation_penalty", 100},
      {"obstacle_activation_distance", 0.02}})
  {
    auto name = "posture." + pair.first;
    if (!node_->has_parameter(name)) {node_->declare_parameter(name, pair.second);}
  }
  for (const auto & pair:std::vector<std::pair<std::string, double>>{
      {"budget", 30}, {"knot_dt", 0.25}, {"max_knots", 60}, {"iterations", 8},
      {"trust_position", 0.08}, {"trust_angle", 0.15}, {"trust_joint", 0.3},
      {"smooth_weight", 1}, {"soft_task_weight", 50}, {"equality_tolerance", 0.02}, {"velocity_fraction", 0.8},
      {"global_layers", 16}, {"global_samples", 32}, {"global_seed", 7}})
  {
    auto name = "posture.nominal." + pair.first;
    if (!node_->has_parameter(name)) {node_->declare_parameter(name, pair.second);}
  }
  if (!node_->has_parameter("posture.nominal.global_fallback")) {
    node_->declare_parameter("posture.nominal.global_fallback", true);
  }
  server_ = rclcpp_action::create_server<Preparation>(
    node_, "task_priority/prepare_cartesian_posture",
    [this](const auto &, std::shared_ptr<const Preparation::Goal> g) {
      bool idle = false;
      if (g->operation > g->APPLY || !busy_.compare_exchange_strong(idle, true)) {
        return rclcpp_action::GoalResponse::REJECT;
      }
      cancel_ = false;
      return rclcpp_action::GoalResponse::ACCEPT_AND_EXECUTE;
    },
    [this](const std::shared_ptr<Handle>) {
      cancel_ = true;
      return rclcpp_action::CancelResponse::ACCEPT;
    },
    [this](const std::shared_ptr<Handle> h) {
      if (worker_.joinable()) {worker_.join();}
      worker_ = std::thread([this, h] {run(h);});
    });
}
PostureAction::~PostureAction()
{
  cancel_ = true;
  if (worker_.joinable()) {worker_.join();}
}
PostureOptions PostureAction::settings() const
{
  const auto value = [this](const char * name, bool zero = false) {
      const auto v = node_->get_parameter(std::string("posture.") + name).as_double();
      if (!std::isfinite(v) || v < 0 || (!zero && v == 0)) {
        throw std::runtime_error(std::string("Invalid posture parameter ") + name);
      }
      return v;
    };
  PostureOptions options{value("budget"), value("radius"), value("duration"), value("settle"),
    value("movement_weight", true), value("tilt_weight", true), value("joint_weight", true), value(
      "clearance_weight", true),
    value("joint_buffer"), value("collision_buffer"),
    value("position_tolerance"), value("orientation_tolerance"), value("joint_tolerance"),
    value("stabilize_weight", true), value("self_collision_weight", true),
    value("activation_penalty"), value("obstacle_activation_distance")};
  if (options.active_penalty <= 1 || options.obstacle_activation_distance >= options.collision_buffer) {
    throw std::runtime_error(
      "posture.activation_penalty must exceed 1 and obstacle_activation_distance must be below collision_buffer");
  }
  return options;
}
PosturePlan PostureAction::nominal_posture_plan(
  StpikSnapshot snapshot, const Preparation::Goal & goal, const StpikOptions & simulation,
  const PostureOptions & options, const NominalPlan & nominal) const
{
  if (nominal.knots.empty()) {throw std::runtime_error("Empty nominal trajectory");}
  PosturePlan p;
  const auto & start = nominal.knots.front();
  const auto joints = snapshot.initial_state.joint_positions.size();
  Eigen::VectorXd current(6 + joints);
  current << snapshot.initial_state.base_position, snapshot.initial_state.base_rpy,
    snapshot.initial_state.joint_positions;
  p.target = snapshot.initial_state;
  p.target.base_position = start.head<3>();
  p.target.base_rpy = start.segment<3>(3);
  p.target.joint_positions = start.tail(joints);
  p.target.joint_velocities = Eigen::VectorXd::Zero(joints);
  p.move = !current.isApprox(start, 1e-10);
  Eigen::VectorXd lower, upper;
  const bool bounded = StpikTaskAccess::joint_bounds(snapshot.tasks, lower, upper) && lower.size() == joints;
  auto & summary = p.summary;
  summary.cost = nominal.cost;
  // The measured posture is not evaluated separately in this mode.
  summary.baseline_feasible = true;
  summary.baseline_cost = p.move ? std::numeric_limits<double>::quiet_NaN() : nominal.cost;
  summary.minimum_clearance = nominal.validation.minimum_clearance;
  summary.minimum_joint_margin = std::numeric_limits<double>::infinity();
  summary.peak_tilt = 0.0;
  for (const auto & knot : nominal.knots) {
    summary.peak_tilt = std::max(summary.peak_tilt, knot.segment<2>(3).norm());
    if (bounded) {
      const Eigen::VectorXd q = knot.tail(joints);
      summary.minimum_joint_margin = std::min(summary.minimum_joint_margin,
        std::min((q - lower).minCoeff(), (upper - q).minCoeff()));
    }
  }
  const Eigen::Isometry3d base = pose_math::base_pose_from_state(p.target);
  summary.configuration.base.header.frame_id = snapshot.model->world_frame();
  summary.configuration.base.pose.position.x = base.translation().x();
  summary.configuration.base.pose.position.y = base.translation().y();
  summary.configuration.base.pose.position.z = base.translation().z();
  const Eigen::Quaterniond q(base.rotation());
  summary.configuration.base.pose.orientation.x = q.x();
  summary.configuration.base.pose.orientation.y = q.y();
  summary.configuration.base.pose.orientation.z = q.z();
  summary.configuration.base.pose.orientation.w = q.w();
  summary.configuration.joint_names = snapshot.model->all_joint_names();
  summary.configuration.positions.assign(p.target.joint_positions.data(),
    p.target.joint_positions.data() + joints);
  p.snapshot = std::move(snapshot);
  p.request = goal;
  p.options = options;
  p.options.equality_tolerance = nominal_settings().equality_tolerance;
  p.simulation = simulation;
  return p;
}
NominalOptions PostureAction::nominal_settings() const
{
  const auto value = [this](const char * name) {
      return node_->get_parameter(std::string("posture.nominal.") + name).as_double();
    };
  NominalOptions options;
  options.budget = value("budget"); options.knot_dt = value("knot_dt");
  options.max_knots = value("max_knots"); options.iterations = value("iterations");
  options.trust_position = value("trust_position"); options.trust_angle = value("trust_angle");
  options.trust_joint = value("trust_joint"); options.smooth_weight = value("smooth_weight");
  options.soft_task_weight = value("soft_task_weight");
  options.equality_tolerance = value("equality_tolerance");
  options.velocity_fraction = value("velocity_fraction");
  options.global_layers = value("global_layers"); options.global_samples = value("global_samples");
  options.global_seed = value("global_seed");
  options.global_fallback = node_->get_parameter("posture.nominal.global_fallback").as_bool();
  std::string reason;
  if (!options.valid(reason)) {throw std::runtime_error("Invalid posture.nominal parameters: " + reason);}
  return options;
}
void PostureAction::fill_nominal(const NominalPlan & plan, Preparation::Result & result) const
{
  result.nominal_joints = plan.joints;
  result.nominal_base = plan.base;
  result.nominal_method = plan.method;
  result.nominal_cost = plan.cost;
}
void PostureAction::run(const std::shared_ptr<Handle> & h)
{
  auto r = std::make_shared<Preparation::Result>();
  auto canceled = [&] {return cancel_.load() || h->is_canceling() || !rclcpp::ok();};
  try {
    const auto & g = *h->get_goal();
    if (g.operation == g.PLAN) {
      plan_.reset();
      StpikSnapshot snapshot;
      {
        std::scoped_lock lock(mutex_);
        if (!ready_() || coordinator_.owns_tasks()) {
          throw std::runtime_error("Controller unavailable or reserved");
        }
        snapshot = StpikTaskAccess::capture(state_, model_, backend_, solver_, manager_.tasks());
      }
      nominal_.reset();
      PosturePlan p;
      if (g.full_trajectory) {
        // The whole-trajectory optimization chooses the initial posture itself (free first
        // knot); the separate posture search, which demands a feasible path without the
        // nominal, is not used.
        const auto simulation = StpikCheckIO::read_options(node_.get());
        const auto options = settings();
        auto nominal = optimize_nominal_trajectory(
          snapshot, g, simulation, options, nominal_settings(), canceled,
          [&](const std::string & stage) {
            auto f = std::make_shared<Preparation::Feedback>(); f->stage = stage;
            h->publish_feedback(f);
          }, nullptr, true);
        p = nominal_posture_plan(std::move(snapshot), g, simulation, options, nominal);
        fill_nominal(nominal, *r);
        nominal_ = std::move(nominal);
      } else {
        p = optimize_posture(
          std::move(snapshot), g, StpikCheckIO::read_options(node_.get()), settings(), canceled,
          [&](unsigned n) {
            auto f = std::make_shared<Preparation::Feedback>();f->stage = "Evaluating whole trajectory";
            f->evaluations = n;
            h->publish_feedback(f);
          });
      }
      p.summary.preparation_id =
        static_cast<uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count());
      plan_ = std::move(p);
      r->proposal = plan_->summary;
    } else {
      if (!plan_ || !g.preparation_id || plan_->summary.preparation_id != g.preparation_id ||
        g.trajectory != plan_->request.trajectory || g.task_id != plan_->request.task_id ||
        g.tip_frame != plan_->request.tip_frame)
      {
        throw std::runtime_error("Unknown or expired posture proposal");
      }
      if (g.full_trajectory && !nominal_) {
        throw std::runtime_error("The posture proposal was planned without a nominal trajectory");
      }
      auto p = std::move(*plan_);
      plan_.reset();
      auto previous = std::move(nominal_);
      nominal_.reset();
      apply(p, g, canceled, !g.full_trajectory);
      r->proposal = p.summary;
      if (g.full_trajectory) {
        // Replan from the settled configuration, warm-started by the PLAN result.
        auto nominal = optimize_nominal_trajectory(
          posture_snapshot(p), g, p.simulation, p.options, nominal_settings(), canceled,
          [&](const std::string & stage) {
            auto f = std::make_shared<Preparation::Feedback>(); f->stage = stage;
            h->publish_feedback(f);
          }, previous ? &*previous : nullptr);
        fill_nominal(nominal, *r);
      }
    }
    r->success = true; r->message = "Posture preparation complete";
  } catch (const std::exception & e) {
    r->success = false;
    r->message = e.what();
  }
  if (canceled()) {r->success = false; r->message = "Posture preparation interrupted";}
  busy_ = false;
  try {
    if (h->is_canceling()) {h->canceled(r);}
    else if (r->success) {h->succeed(r);}
    else {h->abort(r);}
  } catch (const std::exception & error) {
    RCLCPP_DEBUG(node_->get_logger(), "Posture action result delivery failed: %s", error.what());
  }
}
void PostureAction::apply(
  PosturePlan & p, const Preparation::Goal & fresh,
  const StpikCheck::Canceled & canceled, bool check_trajectory)
{
  if (p.request.robot_description != fresh.robot_description ||
    p.request.robot_description_semantic != fresh.robot_description_semantic)
  {
    throw std::runtime_error("Planning model changed");
  }
  if (p.simulation.configuration_key() != StpikCheckIO::read_options(node_.get()).configuration_key()) {
    throw std::runtime_error("STPIK settings changed after posture planning");
  }
  std::unique_lock<std::mutex> lock(mutex_);
  std::string error;
  if (canceled() || !ready_() || coordinator_.owns_tasks() ||
    !StpikTaskAccess::compatible(
      p.snapshot, state_, *model_, backend_.get(), solver_,
      manager_.tasks(), p.simulation, error))
  {
    throw std::runtime_error("Obsolete posture proposal: " + error);
  }
  p.snapshot = StpikTaskAccess::capture(state_, model_, backend_, solver_, manager_.tasks());
  if (!p.snapshot.unavailable_reason.empty()) {
    throw std::runtime_error(p.snapshot.unavailable_reason);
  }
  if (!p.move) {p.target = p.snapshot.initial_state;}
  lock.unlock();
  // Whole-trajectory mode only rechecks the repositioning here; the nominal replanned
  // from the settled posture validates the path with robot_nominal afterwards.
  if (check_trajectory) {recheck_posture(p, fresh, canceled);}
  else {check_repositioning(p, fresh, canceled);}
  lock.lock();
  if (canceled() || !ready_() || coordinator_.owns_tasks() ||
    !StpikTaskAccess::compatible(
      p.snapshot, state_, *model_, backend_.get(), solver_,
      manager_.tasks(), p.simulation, error))
  {
    throw std::runtime_error("Controller changed during posture revalidation: " + error);
  }
  if (!p.move) {return;}
  if (!coordinator_.reserve_posture(p.request.task_id)) {throw std::runtime_error("Controller busy");}
  try {
    const auto tasks = manager_.tasks();
    auto validate_tasks = validate_posture_hierarchy(
      tasks, p.request.task_id, p.simulation.numerical_tolerance, p.options.equality_tolerance);
    validate_tasks(manager_.computations());
    const auto commands = prepare_tasks(tasks, *model_, p.request);
    TaskBase * pose_task = nullptr;
    for (const auto & task : tasks) {if (task->id() == p.request.task_id) {pose_task = task.get();}}
    const auto start = state_;
    int64_t previous = start.ros_time_ns;
    while (true) {
      if (canceled() || !ready_() || !coordinator_.posture_reserved() ||
        manager_.arm_motion_stopped())
      {
        throw std::runtime_error("Posture movement interrupted");
      }
      if (state_.ros_time_ns < previous) {throw std::runtime_error("ROS clock moved backwards");}
      previous = state_.ros_time_ns;
      double time = (previous - start.ros_time_ns) * 1e-9;
      // With an offset on the pose task, the preparation request is expressed for the controlled point.
      auto tip = pose_math::compute_tip_state(*backend_, state_, p.request.tip_frame,
          pose_task);
      if (!tip_within(tip.pose, p.request)) {
        throw std::runtime_error("Tool moved outside preparation tolerance");
      }
      StpikAction::Result safety;
      for (const auto & task:tasks) {
        if (!StpikTaskAccess::safety(*task, state_, p.simulation.numerical_tolerance, safety)) {
          throw std::runtime_error(safety.code + ": " + (safety.message.empty() ?
            "Safety constraint violated: " + safety.task_id + "/" + safety.constraint : safety.message));
        }
      }
      try {
        validate_tasks(manager_.computations());
      } catch (const std::runtime_error & error) {
        throw std::runtime_error(
                "Posture application at t=" + std::to_string(time) + " s: " + error.what());
      }
      posture_reference(commands, *model_, start, p.target, time / p.options.duration,
        p.options.duration);
      if (time >= p.options.duration + p.options.settle) {
        if (!posture_reached(state_, p.target, p.options)) {
          throw std::runtime_error("Posture settling timeout");
        }
        break;
      }
      lock.unlock();
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
      lock.lock();
    }
  } catch (...) {
    if (coordinator_.posture_reserved()) {coordinator_.release(state_, *backend_, error);}
    throw;
  }
  coordinator_.release(state_, *backend_, error, true);
  // The real plant can settle anywhere inside the goal tolerances. Check the path
  // once more from that measured configuration against the execution scene.
  p.snapshot = StpikTaskAccess::capture(state_, model_, backend_, solver_, manager_.tasks());
  if (!p.snapshot.unavailable_reason.empty()) {throw std::runtime_error(p.snapshot.unavailable_reason);}
  p.target = state_;
  lock.unlock();
  if (check_trajectory) {recheck_posture(p, fresh, canceled);}
}
}
