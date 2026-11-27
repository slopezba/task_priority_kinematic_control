#include "task_priority_kinematic_control/core/local_prediction.hpp"
#include "task_priority_kinematic_control/core/posture_scene.hpp"
#include "task_priority_kinematic_control/tasks/robot_nominal_task.hpp"
#include "task_priority_kinematic_control/tasks/frame_pose_task.hpp"
#include "sura_manipulation_actions/progress_nominal.hpp"

#include <optional>
#include <set>

namespace task_priority_kinematic_control
{
namespace
{
namespace sm = sura_manipulation_actions;
using Segment = sm::msg::LocalSegment;
using Clock = std::chrono::steady_clock;
using Vector = Eigen::VectorXd;

geometry_msgs::msg::Pose poseMessage(const Eigen::Isometry3d & pose)
{
  geometry_msgs::msg::Pose out;
  out.position.x = pose.translation().x();
  out.position.y = pose.translation().y();
  out.position.z = pose.translation().z();
  const Eigen::Quaterniond orientation(pose.rotation());
  out.orientation.x = orientation.x();
  out.orientation.y = orientation.y();
  out.orientation.z = orientation.z();
  out.orientation.w = orientation.w();
  return out;
}

struct ExploredPath
{
  sm::msg::LocalCandidatePath path;
  std::optional<StpikSnapshot> endpoint;
  double progress = 0.0;
};

struct Trial
{
  Segment segment;
  StpikAction::Result report;
  Eigen::MatrixXd initial_basis;
  double exploration_cost = 0;
  double first_error = 0.0;
  double cost = std::numeric_limits<double>::infinity();
};

class Search
{
public:
  Search(
    const StpikSnapshot & start, const StpikAction::Goal & goal, const StpikOptions & options,
    const StpikCheck::Canceled & canceled)
  : start_(start), goal_(goal), p_(goal.prediction), options_(options), canceled_(canceled)
  {
    std::string reason;
    if (!start.unavailable_reason.empty() || !start.model || !start.backend) {
      throw std::runtime_error("Snapshot unavailable: " + start.unavailable_reason);
    }
    if (!options.valid(reason) || !path_.initTimed(goal.trajectory, reason) ||
      !global_.init(goal.nominal_joints, goal.nominal_base, p_.global_progress, reason))
    {throw std::invalid_argument(reason);}
    names_ = start.model->all_joint_names();
    if (goal.nominal_joints.joint_names != names_ ||
      goal.nominal_base.header.frame_id != start.model->world_frame())
    {
      throw std::invalid_argument("Global nominal must use controller joint order and world frame");
    }
    for (double limit : {p_.horizon, p_.budget, p_.position_scale, p_.angle_scale}) {
      if (!std::isfinite(limit) || limit <= 0) {
        throw std::invalid_argument("Invalid prediction limits");
      }
    }
    for (double value :
      {p_.progress_rate_scale, p_.retreat_window, p_.gain, p_.amplitude, p_.recovery_threshold,
        p_.improvement})
    {
      if (!std::isfinite(value) || value < 0) {
        throw std::invalid_argument("Invalid recovery parameters");
      }
    }
    for (double w : p_.weights) {
      if (!std::isfinite(w) || w < 0) {throw std::invalid_argument("Invalid prediction weights");}
    }
    if (!(0 <= p_.start_progress && p_.start_progress <= p_.maximum_progress &&
      p_.maximum_progress <= 1) ||
      !std::isfinite(p_.incumbent_elapsed) || p_.incumbent_elapsed < 0 ||
      p_.horizon / options.dt > 10000)
    {throw std::invalid_argument("Invalid prediction horizon or progress");}
    const auto task = find_robot_nominal_task(start.tasks);
    const auto cut = std::find(start.tasks.begin(), start.tasks.end(), task);
    if (!task || !task->enabled()) {
      throw std::invalid_argument("Enabled robot_nominal is required");
    }
    cut_ = cut - start.tasks.begin();
    auto selected = std::find_if(
      start.tasks.begin(), cut, [&](const auto & t) {
        return t->id() == goal.task_id;
      });
    if (selected == cut) {throw std::invalid_argument("Tip task must precede robot_nominal");}
    tip_ = selected - start.tasks.begin();
    if (auto pose = std::dynamic_pointer_cast<FramePoseTask>(*selected)) {
      activation_ = pose->activation();
    }
    const auto & orientation = goal.tolerance_orientation;
    axes_ =
      Eigen::Quaterniond(
      orientation.w, orientation.x, orientation.y,
      orientation.z).toRotationMatrix();
    tolerance_ = Eigen::Map<const sm::BaseCoordinates>(goal.path_tolerance.data());
    scales_ = Vector::Constant(start.model->total_dofs(), p_.angle_scale);
    for (size_t i = 0; i < start.model->base_dofs(); ++i) {
      if (start.model->active_base_dofs()[i] < 3) {scales_(i) = p_.position_scale;}
    }
    if (!p_.incumbent.progress.empty()) {
      if (p_.incumbent.progress.size() < 2 ||
        p_.incumbent.progress.size() != p_.incumbent.joints.points.size() ||
        p_.incumbent.joints.joint_names != names_ ||
        !incumbent_.init(p_.incumbent.joints, p_.incumbent.base, reason))
      {throw std::invalid_argument("Invalid incumbent: " + reason);}
      for (double s : p_.incumbent.progress) {
        if (!(s >= 0 && s <= 1)) {throw std::invalid_argument("Invalid incumbent progress");}
      }
    }
    Preparation::Goal scene_goal;
    scene_goal.robot_description = p_.robot_description;
    scene_goal.robot_description_semantic = p_.robot_description_semantic;
    scene_goal.scene = p_.scene;
    scene_ = std::make_unique<PostureScene>(scene_goal, *start.model);
    for (size_t j = 0; j < names_.size(); ++j) {
      if (scene_->prismatic(j)) {scales_(start.model->base_dofs() + j) = p_.position_scale;}
      if (scene_->continuous(j)) {continuous_.push_back(j);}
    }
    global_.unwrapContinuous(continuous_);
    const double rate = path_.duration() > 0 ? 1.0 / path_.duration() : 0.0;
    advance_ = std::min(1.0, rate * p_.horizon * p_.progress_rate_scale);
  }

  StpikAction::Result run()
  {
    const Segment continuation = reference(1.0, true);
    Trial incumbent = simulate(continuation, nullptr, false);
    const double continuation_cost = incumbent.cost;
    Trial best = std::move(incumbent);
    const bool recovering = best.report.configuration_error > p_.recovery_threshold &&
      best.report.configuration_error >= 0.9 * best.first_error;
    if (p_.explore && (!std::isfinite(best.cost) || recovering ||
      (advance_ > 0 && continuation.progress.back() <= p_.start_progress && p_.start_progress < 1)))
    {
      // Visit nominal first, then both signs in each direction. A physical probe
      // remains fixed while every subsequent step reprojects it into its current basis.
      for (double speed : {1.0, 0.5, 0.0, -0.5}) {
        if (advance_ == 0 && speed != 1.0) {continue;}
        if (expired()) {break;}
        const Segment guide = reference(speed, false);
        auto nominal = simulate(guide, nullptr, true);
        consider(nominal, best);
        for (Eigen::Index i = 0; i < nominal.initial_basis.cols() && !expired(); ++i) {
          for (double sign : {1.0, -1.0}) {
            const Vector direction = sign * p_.amplitude * nominal.initial_basis.col(i);
            auto shot = simulate(guide, &direction, true);
            consider(shot, best);
            if (expired()) {break;}
          }
        }
      }
    }
    auto out = std::move(best.report);
    out.continuation_cost = continuation_cost;
    out.segment_cost = best.cost;
    out.candidates = trials_;
    out.compute_seconds = elapsed();
    if (canceled_ && canceled_()) {
      out.outcome = out.CANCELED; out.code = "CANCELED";
    } else if (std::isfinite(best.cost)) {
      out.outcome = out.SUCCESS; out.code = "SUCCESS"; out.segment = std::move(best.segment);
      out.message = "Kinematic continuation validated by closed-loop replay";
    } else {
      out.outcome = expired() ? out.UNAVAILABLE : out.FAIL;
      out.message = out.code + ": " + out.message;
      out.code = expired() ? "PREDICTION_TIMEOUT" : "NO_LOCAL_CONTINUATION";
    }
    // Draw continuations with a separate budget, after the execution decision.
    // Divide the remaining time among candidates that still need a rollout.
    const auto recovery_deadline = Clock::now() + std::chrono::duration<double>(p_.budget);
    size_t pending = std::count_if(explored_.begin(), explored_.end(),
        [](const auto & candidate) {return candidate.endpoint.has_value();});
    for (auto & candidate : explored_) {
      if (candidate.endpoint) {
        const auto now = Clock::now();
        const auto deadline = now + (recovery_deadline - now) / static_cast<double>(pending--);
        if (now < deadline && !(canceled_ && canceled_())) {
          try {
            recover(candidate, [&] {return Clock::now() >= deadline || (canceled_ && canceled_());});
          } catch (const std::exception &) {
            // Keep the sampled prefix when this visualization cannot continue.
          }
        }
      }
      out.explored_paths.push_back(std::move(candidate.path));
    }
    return out;
  }

private:
  double elapsed() const {return std::chrono::duration<double>(Clock::now() - started_).count();}
  bool expired() const {return (canceled_ && canceled_()) || elapsed() >= p_.budget;}

  Segment knots(
    const std::vector<double> & times, const std::vector<double> & progress,
    const std::vector<sm::BaseCoordinates> & base, const std::vector<Vector> & joints) const
  {
    Segment out;
    std::string reason;
    if (!sm::buildWholeBodyNominal(
        times, base, joints, names_, start_.model->world_frame(),
        out.joints, out.base, reason)) {throw std::runtime_error(reason);}
    out.progress = progress;
    return out;
  }

  Segment reference(double speed, bool follow) const
  {
    const int count = std::max(2, static_cast<int>(std::ceil(p_.horizon / options_.dt)));
    const double low = std::max(0.0, p_.maximum_progress - p_.retreat_window);
    std::vector<double> times, progress;
    std::vector<sm::BaseCoordinates> base;
    std::vector<Vector> joints;
    sm::WholeBodyNominalSample splice;
    std::string error;
    if (!incumbent_.empty() && !incumbent_.sample(p_.incumbent_elapsed, splice, error)) {
      throw std::runtime_error(error);
    }
    for (int i = 0; i <= count; ++i) {
      const double u = static_cast<double>(i) / count, t = p_.horizon * u;
      double s =
        std::clamp(
        p_.start_progress + speed * advance_ * u, std::min(
          low,
          p_.start_progress), 1.0);
      sm::WholeBodyNominalSample sample;
      if (follow && !incumbent_.empty() && p_.incumbent_elapsed + t < incumbent_.duration()) {
        const double local = p_.incumbent_elapsed + t;
        s = sm::segmentPhase(p_.incumbent, local).s;
        std::string error;
        if (!incumbent_.sample(local, sample, error)) {throw std::runtime_error(error);}
      } else {
        if (follow && !incumbent_.empty()) {
          const double extra = p_.incumbent_elapsed + t - incumbent_.duration();
          const double end = p_.incumbent.progress.back();
          s = std::clamp(end + advance_ * extra / p_.horizon, 0.0, 1.0);
        }
        sample = global_.sample(s, 0.0);
      }
      if (!incumbent_.empty() &&
        (i == 0 || (follow && p_.incumbent_elapsed + t >= incumbent_.duration())))
      {
        // Join the previous segment continuously. Replay validates the entire
        // transition back to the global reference and its following interval.
        const double extra = std::max(0.0, p_.incumbent_elapsed + t - incumbent_.duration());
        const double blend = i == 0 ? 0.0 : std::clamp(extra / (0.5 * p_.horizon), 0.0, 1.0);
        sm::WholeBodyNominalSample join = splice;
        if (i && !incumbent_.sample(incumbent_.duration(), join, error)) {
          throw std::runtime_error(error);
        }
        sample.joints = (1 - blend) * join.joints + blend * sample.joints;
        sample.pose.translation() = (1 - blend) * join.pose.translation() + blend *
          sample.pose.translation();
        sample.pose.linear() = Eigen::Quaterniond(join.pose.rotation()).slerp(
          blend, Eigen::Quaterniond(sample.pose.rotation())).toRotationMatrix();
        sample.base = sm::baseCoordinates(sample.pose);
      }
      times.push_back(t); progress.push_back(s); joints.push_back(sample.joints);
      base.push_back(base.empty() ? sample.base : sm::baseCoordinates(sample.pose, base.back()));
    }
    return knots(times, progress, base, joints);
  }

  Vector tangent(
    const sm::WholeBodyNominalSample & sample, const WholeBodyState & state,
    bool velocity) const
  {
    sm::BaseCoordinates base;
    if (velocity) {
      base << sm::baseLinearVector(sample.twist), sm::baseAngularVector(sample.twist);
    } else {
      base = sm::cartesianTrackingError(
        sample.pose, pose_math::base_pose_from_state(
          state), Eigen::Matrix3d::Identity());
    }
    Vector out(scales_.size());
    const auto & axes = start_.model->active_base_dofs();
    for (size_t i = 0; i < axes.size(); ++i) {out(i) = base(axes[i]);}
    out.tail(state.joint_positions.size()) = velocity ? sample.joint_velocities :
      (sample.joints - state.joint_positions).eval();
    if (!velocity) {
      for (auto j : continuous_) {
        const auto index = static_cast<Eigen::Index>(axes.size() + j);
        out(index) = std::remainder(out(index), 2.0 * std::acos(-1.0));
      }
    }
    return out.cwiseQuotient(scales_);
  }

  bool rejoined(double progress, const Eigen::Isometry3d & pose) const
  {
    const auto target = path_.atProgress(progress);
    return (pose.translation() - target.translation()).norm() <= options_.rejoin_position_tolerance &&
      Eigen::Quaterniond(pose.rotation()).angularDistance(Eigen::Quaterniond(target.rotation())) <=
      options_.rejoin_orientation_tolerance;
  }

  void recover(ExploredPath & candidate, const StpikCheck::Canceled & canceled)
  {
    auto & endpoint = *candidate.endpoint;
    const double rate = advance_ / p_.horizon;
    const double traversal = rate > 0 ? (1.0 - candidate.progress) / rate : 0.0;
    const double duration = std::max(p_.horizon, traversal + goal_.goal_time_tolerance);
    const auto progress = [&](double time) {return std::min(1.0, candidate.progress + rate * time);};
    const auto initial_pose = pose_math::base_pose_from_state(endpoint.initial_state);
    const auto initial_base = sm::baseCoordinates(initial_pose);
    const Vector initial_joints = endpoint.initial_state.joint_positions;
    const double blend_time = 0.5 * p_.horizon;
    // Sampling the reference is bounded independently of the integration step.
    const int count = static_cast<int>(std::clamp(std::ceil(duration / options_.dt), 2.0, 10000.0));
    std::vector<double> times, phases;
    std::vector<sm::BaseCoordinates> base;
    std::vector<Vector> joints;
    for (int i = 0; i <= count; ++i) {
      if (canceled()) {return;}
      const double time = duration * (static_cast<double>(i) / count);
      const double s = progress(time);
      auto sample = global_.sample(s, 0.0);
      for (auto j : continuous_) {
        sample.joints(j) = initial_joints(j) +
          std::remainder(sample.joints(j) - initial_joints(j), 2.0 * std::acos(-1.0));
      }
      const double u = std::min(1.0, time / blend_time);
      const double blend = u * u * (3.0 - 2.0 * u);
      auto pose = sample.pose;
      pose.translation() = (1.0 - blend) * initial_pose.translation() + blend * pose.translation();
      pose.linear() = Eigen::Quaterniond(initial_pose.rotation()).slerp(
        blend, Eigen::Quaterniond(sample.pose.rotation())).toRotationMatrix();
      times.push_back(time);
      phases.push_back(s);
      base.push_back(sm::baseCoordinates(pose, base.empty() ? initial_base : base.back()));
      joints.push_back((1.0 - blend) * initial_joints + blend * sample.joints);
    }
    auto ref = knots(times, phases, base, joints);
    auto tracking = goal_;
    tracking.samples.clear();
    tracking.trajectory.duration = duration;
    tracking.goal_time_tolerance = 0.0;
    tracking.nominal_joints = std::move(ref.joints);
    tracking.nominal_base = std::move(ref.base);
    StpikCheck::Rollout rollout;
    rollout.prefix.scales = scales_;
    rollout.target = [&](double time, Eigen::Isometry3d & pose, sm::BaseCoordinates & velocity) {
        const double s = progress(time);
        pose = path_.atProgress(s);
        velocity = path_.progressTangent() * (s < 1.0 ? rate : 0.0);
      };
    rollout.command = [&](double, const WholeBodyState & state,
        const std::vector<TaskComputation> &, const sm::BaseCoordinates &,
        const HierarchySolver::Prefix &, WholeBodyCommand & command) {
        if (!scene_->within_model_limits(
            state, command.generalized_velocity.tail(state.joint_positions.size())))
        {throw std::runtime_error("MoveIt joint position or velocity bound exceeded");}
        if (!(scene_->clearance(state) > 0.0)) {
          throw std::runtime_error(scene_->collision_diagnostic());
        }
      };
    rollout.stop_when = [&](double time, const Eigen::Isometry3d & pose) {
        if (time > 0.0) {candidate.path.poses.push_back(poseMessage(pose));}
        return rejoined(progress(time), pose);
      };
    auto result = StpikCheck::run(
      std::move(endpoint), tracking, options_, canceled, {}, {}, {}, {}, &rollout);
    candidate.path.rejoined = result.outcome == result.SUCCESS;
  }

  Trial simulate(const Segment & ref, const Vector * perturbation, bool shooting)
  {
    Trial trial;
    ++trials_;
    auto snap = StpikTaskAccess::capture(
      start_.initial_state, start_.model, start_.backend,
      start_.solver, start_.tasks);
    snap.captured = start_.captured;
    const auto tasks = snap.tasks;
    auto tracking = goal_;
    tracking.samples.clear();
    tracking.trajectory.duration = p_.horizon;
    tracking.goal_time_tolerance = 0;
    tracking.goal_tolerance = tracking.path_tolerance;
    tracking.nominal_joints = ref.joints; tracking.nominal_base = ref.base;
    StpikCheck::Rollout rollout;
    rollout.prefix.before_task = cut_; rollout.prefix.scales = scales_;
    rollout.target = [&](double time, Eigen::Isometry3d & pose, sm::BaseCoordinates & velocity) {
        const auto phase = sm::segmentPhase(ref, time);
        pose = path_.atProgress(phase.s); velocity = path_.progressTangent() * phase.rate;
      };
    double sum = 0, terminal = 0, terminal_cartesian = 0,
      min_clearance = std::numeric_limits<double>::infinity();
    double last_time = 0, confirmed = p_.start_progress;
    Vector previous;
    size_t rank = 0;
    std::set<std::string> active;
    std::vector<sm::msg::PredictionSample> trace;
    std::vector<double> times, phases;
    std::vector<sm::BaseCoordinates> base;
    std::vector<Vector> joints;
    ExploredPath explored;
    size_t rejoin_sample = 1;
    rollout.observe_pose = [&](double time, const Eigen::Isometry3d & pose) {
        explored.path.poses.push_back(poseMessage(pose));
        if (!rejoined(sm::segmentPhase(ref, time).s, pose)) {
          rejoin_sample = explored.path.poses.size() + 1;
        }
      };
    rollout.command = [&](double time, const WholeBodyState & state,
        const std::vector<TaskComputation> & computations,
        const sm::BaseCoordinates & tip_error, const HierarchySolver::Prefix & prefix,
        WholeBodyCommand & cmd) {
        const auto phase = sm::segmentPhase(ref, time);
        const auto nominal = global_.sample(phase.s, phase.rate);
        const Vector deviation = tangent(nominal, state, false);
        if (time == 0) {trial.initial_basis = prefix.basis; trial.first_error = deviation.norm();}
        rank = prefix.rank;
        const double dt = time - last_time;
        last_time = time;
        confirmed = sm::measuredProgress(
          path_, confirmed, computations[tip_].frame_pose, axes_,
          activation_, tolerance_, 1.0);
        if (phase.s != p_.start_progress &&
          path_.progressTangent().cwiseProduct(activation_).norm() < 1e-12)
        {
          throw std::runtime_error("Progress unobservable on active task axes");
        }
        const auto lead =
          sm::cartesianTrackingError(
          path_.atProgress(phase.s), path_.atProgress(
            confirmed), axes_, activation_);
        if ((lead.cwiseAbs().array() > tolerance_.array() + options_.numerical_tolerance).any()) {
          throw std::runtime_error("Progress exceeds measured tip tracking tube");
        }
        double effort = 0;
        if (shooting) {
          Vector particular = prefix.particular.cwiseQuotient(scales_);
          Vector bias = tangent(nominal, state, true) + p_.gain * deviation - particular;
          if (perturbation) {
            bias += *perturbation;
            effort = (prefix.basis.transpose() * *perturbation).squaredNorm();
          }
          cmd.generalized_velocity =
            scales_.cwiseProduct(particular + prefix.basis * (prefix.basis.transpose() * bias));
          const auto pose = pose_math::base_pose_from_state(state);
          base.push_back(
            base.empty() ? sm::baseCoordinates(pose) : sm::baseCoordinates(
              pose,
              base.back()));
          times.push_back(time); phases.push_back(phase.s); joints.push_back(state.joint_positions);
        }
        if (!scene_->within_model_limits(
            state,
            cmd.generalized_velocity.tail(state.joint_positions.size())))
        {
          throw std::runtime_error("MoveIt joint position or velocity bound exceeded");
        }
        const double clearance = scene_->clearance(state);
        if (!(clearance > 0)) {throw std::runtime_error(scene_->collision_diagnostic());}
        min_clearance = std::min(min_clearance, clearance);
        double margin = 1.0 / (1.0 + clearance / p_.position_scale);
        std::vector<std::string> active_now;
        for (size_t i = 0; i < tasks.size(); ++i) {
          if (!tasks[i]->enabled()) {continue;}
          for (const auto & m : StpikTaskAccess::preventive_margins(*tasks[i], state)) {
            margin += std::pow(std::max(0.0, m.deficit), 2);
          }
          PostureRegion region;
          if (i < cut_ &&
            StpikTaskAccess::posture_region(*tasks[i], region) && !region.always_active)
          {
            const bool outside = region.radial ? region.error.norm() >
              region.limit(0) + options_.numerical_tolerance :
              (region.error.cwiseAbs().array() >
              region.limit.array() + options_.numerical_tolerance).any();
            if (outside) {throw std::runtime_error("Priority region exceeded: " + tasks[i]->id());}
          }
          if (computations[i].active && computations[i].structural_zero_rows &&
            !computations[i].jacobian.isZero())
          {
            active.insert(tasks[i]->id()); active_now.push_back(tasks[i]->id());
          }
        }
        const Vector normalized = cmd.generalized_velocity.cwiseQuotient(scales_);
        double smooth = previous.size() &&
          dt > 0 ? ((normalized - previous) / dt).squaredNorm() : 0;
        previous = normalized;
        terminal = deviation.norm();
        terminal_cartesian = tip_error.cwiseQuotient(tolerance_.cwiseMax(1e-6)).squaredNorm();
        sm::msg::PredictionSample record;
        record.tip_pose = poseMessage(computations[tip_].frame_pose);
        record.elapsed = time; record.progress = phase.s; record.confirmed_progress = confirmed;
        record.hierarchy_rank = rank; record.configuration_error = terminal;
        record.active_constraints = std::move(active_now); record.clearance = clearance;
        std::copy_n(tip_error.data(), 6, record.cartesian_error.begin());
        trace.push_back(std::move(record));
        trial.exploration_cost += dt * p_.weights[5] * effort / p_.horizon;
        sum += dt * (p_.weights[0] * deviation.squaredNorm() + p_.weights[1] * terminal_cartesian +
          p_.weights[3] * margin + p_.weights[4] * smooth + p_.weights[5] * effort);
      };
    auto settings = options_;
    if (!shooting) {settings.dt = std::max(1e-6, 0.5 * settings.dt);}
    WholeBodyState terminal_state;
    trial.report = StpikCheck::run(
      snap, tracking, settings, [this] {return expired();}, {},
      [&](const WholeBodyState & state, double time) {
        if (time >= p_.horizon) {terminal_state = state;}
      }, {}, {}, &rollout);
    trial.report.trace = std::move(trace);
    trial.report.hierarchy_rank = rank;
    trial.report.configuration_error = terminal;
    trial.report.active_constraints.assign(active.begin(), active.end());
    trial.report.minimum_clearance = std::min(min_clearance, trial.report.minimum_clearance);
    if (trial.report.outcome == trial.report.SUCCESS) {
      if (rejoin_sample <= explored.path.poses.size()) {
        explored.path.poses.resize(rejoin_sample);
        explored.path.rejoined = true;
      } else {
        snap.initial_state = std::move(terminal_state);
        snap.captured += std::chrono::duration_cast<Clock::duration>(
          std::chrono::duration<double>(p_.horizon));
        explored.endpoint = std::move(snap);
        explored.progress = ref.progress.back();
      }
      if (shooting && !incumbent_.empty()) {
        sm::WholeBodyNominalSample splice;
        std::string reason;
        if (!incumbent_.sample(p_.incumbent_elapsed, splice, reason)) {
          throw std::runtime_error(reason);
        }
        const sm::BaseCoordinates delta =
          sm::baseCoordinates(splice.pose, base.front()) - base.front();
        const Vector shift = splice.joints - joints.front();
        for (size_t k = 0; k < times.size(); ++k) {
          const double remaining = std::max(0.0, 1.0 - 2 * times[k] / p_.horizon);
          const double blend = remaining * remaining * (3.0 - 2.0 * remaining);
          base[k] += blend * delta; joints[k] += blend * shift;
        }
      }
      trial.segment = shooting ? knots(times, phases, base, joints) : ref;
      const double missed = std::min(1.0, p_.start_progress + advance_) - ref.progress.back();
      trial.cost = sum / p_.horizon + p_.weights[2] *
        (terminal * terminal + terminal_cartesian + missed * missed);
    }
    explored_.push_back(std::move(explored));
    return trial;
  }

  void consider(const Trial & shot, Trial & best)
  {
    if (!std::isfinite(shot.cost) || expired()) {return;}
    auto replay = simulate(shot.segment, nullptr, false);
    replay.cost += shot.exploration_cost;
    if (!std::isfinite(replay.cost)) {return;}
    // Exploration energy also belongs to the reference being ranked on replay.
    if (std::isfinite(best.cost) && !(replay.cost < best.cost - p_.improvement)) {return;}
    best = std::move(replay);
  }

  const StpikSnapshot & start_;
  const StpikAction::Goal & goal_;
  const sm::msg::LocalPrediction & p_;
  StpikOptions options_;
  StpikCheck::Canceled canceled_;
  Clock::time_point started_ = Clock::now();
  sm::CartesianPosePath path_;
  sm::ProgressNominal global_;
  sm::WholeBodyNominal incumbent_;
  std::unique_ptr<PostureScene> scene_;
  size_t cut_ = 0, tip_ = 0;
  unsigned trials_ = 0;
  std::vector<ExploredPath> explored_;
  std::vector<std::string> names_;
  std::vector<size_t> continuous_;
  Vector scales_;
  sm::BaseCoordinates activation_ = sm::BaseCoordinates::Ones(), tolerance_;
  Eigen::Matrix3d axes_;
  double advance_ = 0;
};
}

StpikAction::Result predict_local_segment(
  const StpikSnapshot & start, const StpikAction::Goal & goal,
  const StpikOptions & options, const StpikCheck::Canceled & canceled)
{
  try {
    return Search(start, goal, options, canceled).run();
  } catch (const std::exception & e) {
    StpikAction::Result out; out.outcome = out.UNAVAILABLE; out.code = "PREDICTION_UNAVAILABLE";
    out.message = e.what(); return out;
  }
}
}
