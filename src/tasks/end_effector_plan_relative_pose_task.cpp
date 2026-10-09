#include "task_priority_kinematic_control/tasks/end_effector_plan_relative_pose_task.hpp"

#include <pluginlib/class_list_macros.hpp>

#include <Eigen/SVD>

#include <algorithm>
#include <array>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <deque>
#include <fstream>
#include <future>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <random>
#include <rclcpp/logging.hpp>
#include <sstream>

namespace task_priority_kinematic_control
{
namespace
{

Eigen::Vector3d quaternion_error(
  const Eigen::Quaterniond & target,
  const Eigen::Quaterniond & current)
{
  const Eigen::Quaterniond q_err = target * current.conjugate();
  Eigen::AngleAxisd aa(q_err);
  return aa.axis() * aa.angle();
}

Eigen::Isometry3d pose_goal_to_isometry(const geometry_msgs::msg::PoseStamped & goal)
{
  Eigen::Isometry3d pose = Eigen::Isometry3d::Identity();
  pose.translation() = Eigen::Vector3d(
    goal.pose.position.x,
    goal.pose.position.y,
    goal.pose.position.z);
  const Eigen::Quaterniond q(
    goal.pose.orientation.w,
    goal.pose.orientation.x,
    goal.pose.orientation.y,
    goal.pose.orientation.z);
  pose.linear() = q.normalized().toRotationMatrix();
  return pose;
}

double duration_to_seconds(const builtin_interfaces::msg::Duration & duration)
{
  return static_cast<double>(duration.sec) + static_cast<double>(duration.nanosec) * 1.0e-9;
}

std::string sanitized_node_name(const std::string & id)
{
  std::string name = "ee_plan_relative_pose_external_planner";
  if (!id.empty()) {
    name += "_" + id;
  }
  for (auto & character : name) {
    const unsigned char c = static_cast<unsigned char>(character);
    if (!std::isalnum(c) && character != '_') {
      character = '_';
    }
  }
  return name;
}

std::vector<double> activation_values_from_parameters(
  const std::map<std::string, rclcpp::Parameter> & parameters)
{
  const auto it = parameters.find("activation");
  if (it == parameters.end()) {
    return {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
  }
  if (it->second.get_type() == rclcpp::ParameterType::PARAMETER_INTEGER_ARRAY) {
    const auto values = it->second.as_integer_array();
    return std::vector<double>(values.begin(), values.end());
  }
  if (it->second.get_type() == rclcpp::ParameterType::PARAMETER_BOOL_ARRAY) {
    const auto values = it->second.as_bool_array();
    std::vector<double> out;
    out.reserve(values.size());
    for (const bool value : values) {
      out.push_back(value ? 1.0 : 0.0);
    }
    return out;
  }
  return it->second.as_double_array();
}

Eigen::MatrixXd damped_pseudoinverse(const Eigen::MatrixXd & matrix, double damping)
{
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    return Eigen::MatrixXd::Zero(matrix.cols(), matrix.rows());
  }
  const Eigen::MatrixXd identity = Eigen::MatrixXd::Identity(matrix.rows(), matrix.rows());
  return matrix.transpose() * (matrix * matrix.transpose() + damping * damping * identity).inverse();
}

double smallest_singular_value(const Eigen::MatrixXd & matrix)
{
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    return 0.0;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
  return svd.singularValues().size() == 0 ? 0.0 : svd.singularValues().tail<1>()(0);
}

double manipulability_measure(const Eigen::MatrixXd & matrix)
{
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    return 0.0;
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix);
  double product = 1.0;
  for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i) {
    product *= std::max(0.0, svd.singularValues()(i));
  }
  return product;
}

int numerical_rank(const Eigen::VectorXd & singular_values, int rows, int cols)
{
  if (singular_values.size() == 0) {
    return 0;
  }
  const double tolerance =
    std::numeric_limits<double>::epsilon() *
    static_cast<double>(std::max(rows, cols)) *
    singular_values(0);
  int rank = 0;
  for (Eigen::Index i = 0; i < singular_values.size(); ++i) {
    if (singular_values(i) > tolerance) {
      ++rank;
    }
  }
  return rank;
}

Eigen::MatrixXd nullspace_basis(const Eigen::MatrixXd & matrix)
{
  if (matrix.rows() == 0 || matrix.cols() == 0) {
    return Eigen::MatrixXd::Identity(matrix.cols(), matrix.cols());
  }
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(matrix, Eigen::ComputeFullV);
  const double tolerance =
    std::numeric_limits<double>::epsilon() *
    static_cast<double>(std::max(matrix.rows(), matrix.cols())) *
    (svd.singularValues().size() > 0 ? svd.singularValues()(0) : 1.0);
  Eigen::Index rank = 0;
  for (Eigen::Index i = 0; i < svd.singularValues().size(); ++i) {
    if (svd.singularValues()(i) > tolerance) {
      ++rank;
    }
  }
  if (rank >= matrix.cols()) {
    return Eigen::MatrixXd::Zero(matrix.cols(), 0);
  }
  return svd.matrixV().rightCols(matrix.cols() - rank);
}

Eigen::VectorXd clamp_to_limits(
  const Eigen::VectorXd & q,
  const Eigen::VectorXd & lower,
  const Eigen::VectorXd & upper)
{
  Eigen::VectorXd out = q;
  for (Eigen::Index i = 0; i < out.size(); ++i) {
    if (i < lower.size() && std::isfinite(lower(i))) {
      out(i) = std::max(out(i), lower(i));
    }
    if (i < upper.size() && std::isfinite(upper(i))) {
      out(i) = std::min(out(i), upper(i));
    }
  }
  return out;
}

Eigen::VectorXd limits_from_parameters(
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const std::string & values_parameter_name,
  const std::vector<std::string> & model_joint_names,
  double default_value)
{
  const auto default_values = std::vector<double>(model_joint_names.size(), default_value);
  auto values = default_values;
  const auto values_it = parameters.find(values_parameter_name);
  if (values_it != parameters.end()) {
    values = values_it->second.as_double_array();
  }
  Eigen::VectorXd limits = Eigen::VectorXd::Constant(model_joint_names.size(), default_value);

  std::vector<std::string> configured_joint_names;
  const auto names_it = parameters.find("joint_names");
  if (names_it != parameters.end()) {
    configured_joint_names = names_it->second.as_string_array();
  }
  if (configured_joint_names.size() == values.size() && !configured_joint_names.empty()) {
    std::map<std::string, double> values_by_joint;
    for (size_t i = 0; i < configured_joint_names.size(); ++i) {
      values_by_joint[configured_joint_names[i]] = values[i];
    }

    for (size_t i = 0; i < model_joint_names.size(); ++i) {
      const auto it = values_by_joint.find(model_joint_names[i]);
      if (it != values_by_joint.end()) {
        limits(static_cast<Eigen::Index>(i)) = it->second;
      }
    }
    return limits;
  }

  for (size_t i = 0; i < model_joint_names.size() && i < values.size(); ++i) {
    limits(static_cast<Eigen::Index>(i)) = values[i];
  }
  return limits;
}

double relaxed_or_specific_tolerance(
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const std::string & name,
  double legacy_default,
  double relaxed_default,
  bool has_relaxed_default)
{
  const auto it = parameters.find(name);
  if (it == parameters.end()) {
    return has_relaxed_default ? relaxed_default : legacy_default;
  }

  const double value = it->second.as_double();
  if (has_relaxed_default && std::abs(value - legacy_default) < 1.0e-12) {
    return relaxed_default;
  }
  return value;
}

double normalized_quality(double value, double scale)
{
  if (!std::isfinite(value) || !std::isfinite(scale) || scale <= 1.0e-12) {
    return 0.0;
  }
  return std::clamp(value / scale, 0.0, 1.0);
}

template<typename Derived>
std::string vector_to_string(const Eigen::MatrixBase<Derived> & vector)
{
  std::ostringstream stream;
  stream << std::setprecision(9) << "[";
  for (Eigen::Index i = 0; i < vector.size(); ++i) {
    if (i > 0) {
      stream << ", ";
    }
    stream << vector(i);
  }
  stream << "]";
  return stream.str();
}

int least_significant_zero_bit(std::uint32_t value)
{
  int bit = 1;
  while ((value & 1U) != 0U && bit < 32) {
    value >>= 1U;
    ++bit;
  }
  return bit;
}

std::vector<std::array<std::uint32_t, 32>> sobol_direction_numbers(std::size_t dimensions)
{
  constexpr std::size_t kBits = 32;
  std::vector<std::array<std::uint32_t, kBits>> directions(dimensions);
  for (auto & direction : directions) {
    direction.fill(0U);
  }
  if (dimensions == 0) {
    return directions;
  }

  for (std::size_t bit = 1; bit <= kBits; ++bit) {
    directions[0][bit - 1] = 1U << (kBits - bit);
  }

  struct SobolParams
  {
    int degree;
    std::uint32_t polynomial;
    std::array<std::uint32_t, 5> initial;
  };
  const std::array<SobolParams, 7> params = {{
      {1, 0U, {1U, 0U, 0U, 0U, 0U}},
      {2, 1U, {1U, 3U, 0U, 0U, 0U}},
      {3, 1U, {1U, 3U, 1U, 0U, 0U}},
      {3, 2U, {1U, 1U, 1U, 0U, 0U}},
      {4, 1U, {1U, 3U, 5U, 13U, 0U}},
      {4, 4U, {1U, 1U, 5U, 5U, 0U}},
      {5, 2U, {1U, 1U, 5U, 5U, 17U}},
    }};

  for (std::size_t dim = 1; dim < dimensions; ++dim) {
    const auto & param = params[std::min(dim - 1, params.size() - 1)];
    const int degree = param.degree;
    for (int bit = 1; bit <= degree; ++bit) {
      directions[dim][bit - 1] = param.initial[bit - 1] << (kBits - bit);
    }
    for (std::size_t bit = static_cast<std::size_t>(degree + 1); bit <= kBits; ++bit) {
      std::uint32_t value =
        directions[dim][bit - static_cast<std::size_t>(degree) - 1] ^
        (directions[dim][bit - static_cast<std::size_t>(degree) - 1] >> degree);
      for (int k = 1; k < degree; ++k) {
        if (((param.polynomial >> static_cast<std::uint32_t>(degree - 1 - k)) & 1U) != 0U) {
          value ^= directions[dim][bit - static_cast<std::size_t>(k) - 1];
        }
      }
      directions[dim][bit - 1] = value;
    }
  }
  return directions;
}

}  // namespace

EndEffectorPlanRelativePoseTask::~EndEffectorPlanRelativePoseTask()
{
  stop_planner();
  if (external_planner_executor_) {
    external_planner_executor_->cancel();
  }
  if (external_planner_spin_thread_.joinable()) {
    external_planner_spin_thread_.join();
  }
}

void EndEffectorPlanRelativePoseTask::configure(
  const std::string & id,
  const std::string & plugin_name,
  const std::map<std::string, rclcpp::Parameter> & parameters,
  const TaskContext & context)
{
  configure_common(id, plugin_name, parameters, context);
  reference_frame_ = get_string_param(parameters, "reference_frame", context.model->left_tip_frame());
  controlled_frame_ = get_string_param(parameters, "controlled_frame", context.model->right_tip_frame());
  joint_names_ = context.model->all_joint_names();

  const auto gains = get_double_array_param(parameters, "gain", {1.0, 1.0, 1.0, 0.7, 0.7, 0.7});
  for (size_t i = 0; i < 6 && i < gains.size(); ++i) {
    gains_(static_cast<Eigen::Index>(i)) = gains[i];
  }

  const auto activation = activation_values_from_parameters(parameters);
  for (size_t i = 0; i < 6 && i < activation.size(); ++i) {
    activation_(static_cast<Eigen::Index>(i)) = activation[i] != 0.0;
  }

  planning_timeout_ = get_double_param(parameters, "planning_timeout", 5.0);
  move_gain_ = get_double_param(parameters, "move_gain", 0.8);
  move_goal_tolerance_ = get_double_param(parameters, "move_goal_tolerance", 0.04);
  blend_duration_ = get_double_param(parameters, "blend_duration", 2.0);
  max_joint_velocity_ = get_double_param(parameters, "max_joint_velocity", 0.25);
  min_sigma_ = get_double_param(parameters, "min_sigma", 0.03);
  min_joint_margin_ = get_double_param(parameters, "min_joint_margin", 0.02);
  min_collision_margin_ = get_double_param(parameters, "min_collision_margin", 0.04);
  nullspace_probe_radius_ = get_double_param(parameters, "nullspace_probe_radius", 0.12);
  nullspace_probe_steps_ = static_cast<int>(get_double_param(parameters, "nullspace_probe_steps", 5.0));
  reprojection_iterations_ =
    static_cast<int>(get_double_param(parameters, "reprojection_iterations", 20.0));
  const bool has_relative_pose_tolerance = parameters.find("relative_pose_tolerance") != parameters.end();
  relative_pose_tolerance_ = get_double_param(parameters, "relative_pose_tolerance", 1.0e-4);
  const double relaxed_tolerance_default =
    has_relative_pose_tolerance ? relative_pose_tolerance_ : 1.0e-4;
  reprojection_tolerance_ =
    relaxed_or_specific_tolerance(
    parameters,
    "reprojection_tolerance",
    1.0e-4,
    relaxed_tolerance_default,
    has_relative_pose_tolerance);

  weight_sigma_ = get_double_param(parameters, "weights.sigma", 1.0);
  weight_collision_ = get_double_param(parameters, "weights.collision", 2.0);
  weight_limits_ = get_double_param(parameters, "weights.limits", 1.0);
  weight_room_ = get_double_param(parameters, "weights.room", 1.5);
  weight_smoothness_ = get_double_param(parameters, "weights.smoothness", 0.5);
  weight_current_distance_ = get_double_param(parameters, "weights.current_distance", 1.2);
  weight_curvature_ = get_double_param(parameters, "weights.curvature", 0.5);
  weight_boundary_ = get_double_param(parameters, "weights.boundary", 0.15);
  weight_support_ = get_double_param(parameters, "weights.support", 0.15);
  weight_relative_error_ = get_double_param(parameters, "weights.relative_error", 5.0);
  debug_ = get_bool_param(parameters, "debug", false);
  fallback_to_servo_on_plan_failure_ =
    get_bool_param(parameters, "fallback_to_servo_on_plan_failure", true);
  global_search_enabled_ = get_bool_param(parameters, "global_search.enabled", false);
  global_random_starts_ =
    static_cast<int>(get_double_param(parameters, "global_search.random_starts", 128.0));
  global_local_perturbation_starts_ =
    static_cast<int>(get_double_param(parameters, "global_search.local_perturbation_starts", 12.0));
  global_seed_ =
    static_cast<unsigned int>(get_double_param(parameters, "global_search.seed", 7.0));
  global_projection_iterations_ =
    static_cast<int>(get_double_param(parameters, "global_search.projection_iterations", 20.0));
  global_projection_tolerance_ =
    relaxed_or_specific_tolerance(
    parameters,
    "global_search.projection_tolerance",
    1.0e-4,
    relaxed_tolerance_default,
    has_relative_pose_tolerance);
  global_solution_tolerance_ =
    relaxed_or_specific_tolerance(
    parameters,
    "global_search.solution_tolerance",
    5.0e-4,
    has_relative_pose_tolerance ? relative_pose_tolerance_ : 5.0e-4,
    has_relative_pose_tolerance);
  global_explore_threshold_ =
    relaxed_or_specific_tolerance(
    parameters,
    "global_search.explore_threshold",
    1.0e-4,
    has_relative_pose_tolerance ? relative_pose_tolerance_ : global_projection_tolerance_,
    has_relative_pose_tolerance);
  global_deduplicate_radius_ =
    get_double_param(parameters, "global_search.deduplicate_radius", 0.004);
  global_connect_radius_ =
    get_double_param(parameters, "global_search.connect_radius", 1.25);
  global_max_edge_projection_distance_ =
    get_double_param(parameters, "global_search.max_edge_projection_distance", 0.08);
  global_max_edge_neighbors_ =
    static_cast<int>(get_double_param(parameters, "global_search.max_edge_neighbors", 2.0));
  global_path_check_steps_ =
    static_cast<int>(get_double_param(parameters, "global_search.path_check_steps", 2.0));
  global_max_components_to_plan_ =
    static_cast<int>(get_double_param(parameters, "global_search.max_components_to_plan", 2.0));
  global_polar_angular_samples_ =
    static_cast<int>(get_double_param(parameters, "global_search.polar_angular_samples", 8.0));
  global_polar_radial_step_ =
    get_double_param(parameters, "global_search.polar_radial_step", 0.06);
  global_polar_max_radius_ =
    get_double_param(parameters, "global_search.polar_max_radius", 0.36);
  global_max_charts_per_component_ =
    static_cast<int>(get_double_param(parameters, "global_search.max_charts_per_component", 3.0));
  global_max_projector_change_ =
    get_double_param(parameters, "global_search.max_projector_change", 0.35);
  global_atlas_max_charts_per_seed_ =
    static_cast<int>(get_double_param(parameters, "global_search.atlas_max_charts_per_seed", 3.0));
  global_atlas_max_total_charts_ =
    static_cast<int>(get_double_param(parameters, "global_search.atlas_max_total_charts", 160.0));
  global_atlas_step_ =
    get_double_param(parameters, "global_search.atlas_step", 0.08);
  global_atlas_neighbor_radius_ =
    get_double_param(parameters, "global_search.atlas_neighbor_radius", 0.12);
  global_projection_trace_enabled_ =
    get_bool_param(parameters, "global_search.projection_trace.enabled", true);
  global_projection_trace_seed_index_ =
    static_cast<int>(get_double_param(parameters, "global_search.projection_trace.seed_index", 0.0));
  global_debug_dump_path_ = get_string_param(
    parameters,
    "global_search.debug_dump_path",
    "/tmp/end_effector_plan_relative_pose_manifold_debug.txt");
  preferred_joint_margin_ = get_double_param(parameters, "preferred_joint_margin", 0.08);
  external_planner_enabled_ = get_bool_param(parameters, "external_planner.enabled", false);
  external_planner_action_name_ = get_string_param(
    parameters,
    "external_planner.action_name",
    "/cirtesub/manipulation/plan_bimanual_joint_trajectory");
  external_planner_wait_timeout_ =
    get_double_param(parameters, "external_planner.wait_timeout", 1.0);
  external_planner_result_timeout_ =
    get_double_param(parameters, "external_planner.result_timeout", 5.0);
  external_planner_goal_tolerance_ =
    relaxed_or_specific_tolerance(
    parameters,
    "external_planner.goal_tolerance",
    0.02,
    has_relative_pose_tolerance ? relative_pose_tolerance_ : 0.02,
    has_relative_pose_tolerance);

  if (external_planner_enabled_) {
    external_planner_node_ = std::make_shared<rclcpp::Node>(
      sanitized_node_name(id_),
      rclcpp::NodeOptions().allow_undeclared_parameters(true));
    external_planner_executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>();
    external_planner_executor_->add_node(external_planner_node_);
    external_planner_client_ =
      rclcpp_action::create_client<PlanBimanualJointTrajectory>(
      external_planner_node_,
      external_planner_action_name_);
    external_planner_spin_thread_ = std::thread{
      [this]() {
        external_planner_executor_->spin();
      }};
  }

  lower_limits_ = limits_from_parameters(
    parameters,
    "lower_limits",
    joint_names_,
    -std::numeric_limits<double>::infinity());
  upper_limits_ = limits_from_parameters(
    parameters,
    "upper_limits",
    joint_names_,
    std::numeric_limits<double>::infinity());

  current_joint_target_ = Eigen::VectorXd::Zero(joint_names_.size());
}

TaskComputation EndEffectorPlanRelativePoseTask::update(
  const WholeBodyState & state,
  const KinematicsBackend & backend)
{
  if (!enabled_) {
    TaskComputation computation;
    computation.status_message = "disabled";
    return computation;
  }

  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (mode_ == Mode::kNeedsPlan) {
      if (!state.joints_valid || state.joint_positions.size() != static_cast<Eigen::Index>(joint_names_.size())) {
        TaskComputation computation;
        computation.status_message = "waiting_for_joints";
        return computation;
      }
      const FrameState reference = backend.get_frame_state(reference_frame_);
      const FrameState controlled = backend.get_frame_state(controlled_frame_);
      PlannerInput input;
      input.state_template = state;
      input.kinematics = backend.clone();
      input.q_current = state.joint_positions;
      input.target_relative_pose = target_relative_pose_;
      input.current_relative_pose = reference.pose.inverse() * controlled.pose;
      input.relative_jacobian = relative_jacobian(reference, controlled);
      start_planning_locked(input);
    }

    if (planner_result_ready_) {
      planner_result_ready_ = false;
      planner_running_ = false;
      if (planner_thread_.joinable()) {
        planner_thread_.join();
      }
      if (planner_result_.success) {
        active_trajectory_ = planner_result_.trajectory;
        current_joint_target_ = active_trajectory_.start;
        trajectory_start_ = std::chrono::steady_clock::now();
        last_plan_score_ = planner_result_.score;
        mode_ = Mode::kMoveToManifold;
        status_message_ = "moving_to_manifold";
      } else {
        status_message_ = planner_result_.timed_out ? "planning_timeout" : planner_result_.message;
        if (!planner_result_.timed_out && fallback_to_servo_on_plan_failure_) {
          debug_log("planning failed, falling back to direct relative-pose servo: " + status_message_);
          mode_ = Mode::kServoRelativePose;
          status_message_ = "servo_relative_pose";
        } else {
          mode_ = Mode::kIdle;
        }
      }
    }
  }

  Mode mode;
  {
    std::lock_guard<std::mutex> lock(mutex_);
    mode = mode_;
  }

  if (mode == Mode::kPlanning || mode == Mode::kIdle || mode == Mode::kNeedsPlan) {
    TaskComputation computation;
    computation.active = false;
    computation.status_message = mode == Mode::kPlanning ? "planning" : status_message_;
    return computation;
  }
  if (mode == Mode::kMoveToManifold) {
    return move_to_manifold_update(state, true);
  }
  if (mode == Mode::kBlendToRelativePose) {
    return blend_update(state, backend);
  }
  return relative_pose_update(backend, 1.0);
}

bool EndEffectorPlanRelativePoseTask::set_pose_goal(const geometry_msgs::msg::PoseStamped & goal)
{
  stop_planner();
  std::lock_guard<std::mutex> lock(mutex_);
  target_relative_pose_ = pose_goal_to_isometry(goal);
  has_target_relative_pose_ = true;
  pose_goal_ = goal;
  has_pose_goal_ = true;
  mode_ = Mode::kNeedsPlan;
  status_message_ = "needs_plan";
  if (debug_) {
    std::ostringstream stream;
    stream << "pose received target_xyz=["
           << goal.pose.position.x << ", "
           << goal.pose.position.y << ", "
           << goal.pose.position.z << "] target_quat_xyzw=["
           << goal.pose.orientation.x << ", "
           << goal.pose.orientation.y << ", "
           << goal.pose.orientation.z << ", "
           << goal.pose.orientation.w << "]";
    debug_log(stream.str());
  }
  return true;
}

bool EndEffectorPlanRelativePoseTask::set_gain(
  const std::vector<double> & gain,
  std::string & message)
{
  if (gain.size() != 6) {
    message = "Plan relative pose gain must contain 6 values";
    return false;
  }
  for (size_t i = 0; i < gain.size(); ++i) {
    if (gain[i] < 0.0) {
      message = "Plan relative pose gain values must be non-negative";
      return false;
    }
    gains_(static_cast<Eigen::Index>(i)) = gain[i];
  }
  message = "Plan relative pose gain updated";
  return true;
}

void EndEffectorPlanRelativePoseTask::reset()
{
  stop_planner();
  std::lock_guard<std::mutex> lock(mutex_);
  mode_ = Mode::kIdle;
  status_message_ = "idle";
}

msg::TaskStatus EndEffectorPlanRelativePoseTask::build_status() const
{
  auto status = TaskBaseCommon::build_status();
  std::lock_guard<std::mutex> lock(mutex_);
  status.active = enabled_ && (
    mode_ == Mode::kMoveToManifold || mode_ == Mode::kBlendToRelativePose ||
    mode_ == Mode::kServoRelativePose);
  status.status_message = status_message_;
  status.target_type = "pose";
  status.joint_names = joint_names_;
  return status;
}

std::vector<double> EndEffectorPlanRelativePoseTask::current_target() const
{
  const Eigen::Quaterniond q(target_relative_pose_.rotation());
  return {
    target_relative_pose_.translation().x(),
    target_relative_pose_.translation().y(),
    target_relative_pose_.translation().z(),
    q.x(),
    q.y(),
    q.z(),
    q.w()};
}

void EndEffectorPlanRelativePoseTask::stop_planner()
{
  planner_cancel_.store(true);
  if (planner_thread_.joinable()) {
    planner_thread_.join();
  }
  planner_cancel_.store(false);
  std::lock_guard<std::mutex> lock(mutex_);
  planner_running_ = false;
  planner_result_ready_ = false;
}

void EndEffectorPlanRelativePoseTask::start_planning_locked(const PlannerInput & input)
{
  if (planner_running_) {
    return;
  }
  if (planner_thread_.joinable()) {
    planner_thread_.join();
  }
  planner_cancel_.store(false);
  planner_running_ = true;
  planner_result_ready_ = false;
  mode_ = Mode::kPlanning;
  status_message_ = "planning";
  debug_log("checking manifolds");
  planner_thread_ = std::thread(&EndEffectorPlanRelativePoseTask::planner_main, this, input);
}

void EndEffectorPlanRelativePoseTask::planner_main(PlannerInput input)
{
  const auto deadline = std::chrono::steady_clock::now() +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(planning_timeout_));
  auto result = plan(input, deadline);
  std::lock_guard<std::mutex> lock(mutex_);
  planner_result_ = std::move(result);
  planner_result_ready_ = true;
}

EndEffectorPlanRelativePoseTask::PlannerResult EndEffectorPlanRelativePoseTask::plan(
  const PlannerInput & input,
  const std::chrono::steady_clock::time_point & deadline)
{
  PlannerResult result;
  auto candidates = explore_candidates(input, deadline);
  if (debug_) {
    std::ostringstream summary;
    int component_count = 0;
    for (const auto & candidate : candidates) {
      component_count = std::max(component_count, candidate.component_id + 1);
    }
    summary << "total candidates found=" << candidates.size()
            << " components found=" << component_count;
    debug_log(summary.str());
    constexpr size_t kMaxDebugCandidates = 20;
    const size_t candidate_log_count = std::min(kMaxDebugCandidates, candidates.size());
    for (size_t i = 0; i < candidate_log_count; ++i) {
      const auto & candidate = candidates[i];
      std::ostringstream stream;
      stream << "candidate C" << i
             << " component=" << candidate.component_id
             << " support=" << candidate.component_sample_count
             << " score=" << candidate.score
             << " component_score=" << candidate.component_score
             << " rel_error=" << candidate.relative_error_norm
             << " sigma=" << candidate.sigma_min
             << " manipulability=" << candidate.manipulability
             << " collision=" << candidate.collision_margin
             << " limits=" << candidate.joint_margin
             << " room=" << candidate.nullspace_room
             << " local_area=" << candidate.local_area
             << " boundary_radius=" << candidate.boundary_radius
             << " smoothness=" << candidate.smoothness
             << " current_distance=" << candidate.current_distance
             << " curvature=" << candidate.curvature;
      debug_log(stream.str());
    }
    if (candidates.size() > candidate_log_count) {
      debug_log(
        "candidate log truncated after " + std::to_string(candidate_log_count) +
        " of " + std::to_string(candidates.size()) + " candidates");
    }
  }
  if (planner_cancel_.load()) {
    result.message = "planning_canceled";
    return result;
  }
  if (std::chrono::steady_clock::now() >= deadline) {
    result.timed_out = true;
    result.message = "planning_timeout";
    return result;
  }
  auto best = std::max_element(
    candidates.begin(), candidates.end(), [](const Candidate & lhs, const Candidate & rhs) {
      return lhs.score < rhs.score;
    });
  if (best == candidates.end()) {
    result.message = "no_valid_manifold_candidate";
    return result;
  }

  std::sort(
    candidates.begin(),
    candidates.end(),
    [](const Candidate & lhs, const Candidate & rhs) {
      return lhs.score > rhs.score;
    });

  std::vector<int> planned_components;
  std::string last_rejection = "no_collision_free_trajectory";
  for (const auto & candidate : candidates) {
    if (planner_cancel_.load()) {
      result.message = "planning_canceled";
      return result;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      result.timed_out = true;
      result.message = "planning_timeout";
      return result;
    }
    if (candidate.component_id >= 0 &&
      std::find(planned_components.begin(), planned_components.end(), candidate.component_id) !=
      planned_components.end())
    {
      continue;
    }
    if (candidate.component_id >= 0) {
      planned_components.push_back(candidate.component_id);
      if (static_cast<int>(planned_components.size()) > std::max(1, global_max_components_to_plan_)) {
        break;
      }
    }

    {
      std::ostringstream stream;
      stream << "planning trajectory to component=" << candidate.component_id
             << " candidate_score=" << candidate.score
             << " component_score=" << candidate.component_score;
      debug_log(stream.str());
    }

    PlannedTrajectory trajectory;
    bool accepted = false;
    bool timed_out = false;
    if (external_planner_enabled_) {
      std::string message;
      accepted = request_external_trajectory(
        input.q_current,
        candidate.q,
        deadline,
        trajectory,
        message,
        timed_out);
      if (!accepted) {
        last_rejection = message;
        debug_log("external trajectory rejected: " + message);
        if (timed_out) {
          result.timed_out = true;
          result.message = "planning_timeout";
          return result;
        }
        continue;
      }
    } else {
      accepted = validate_trajectory(input.q_current, candidate.q);
      if (!accepted) {
        last_rejection = "local_trajectory_rejected";
        debug_log("trajectory rejected");
        continue;
      }
      trajectory.start = input.q_current;
      trajectory.goal = candidate.q;
      const double distance = (candidate.q - input.q_current).cwiseAbs().maxCoeff();
      trajectory.duration = std::max(0.1, distance / std::max(1.0e-6, max_joint_velocity_));
    }

    result.success = true;
    result.q_selected = candidate.q;
    result.trajectory = std::move(trajectory);
    result.score = candidate.score;
    result.message = "plan_ready";
    debug_log("trajectory accepted");
    return result;
  }

  result.message = last_rejection;
  return result;
}

std::vector<EndEffectorPlanRelativePoseTask::Candidate>
EndEffectorPlanRelativePoseTask::explore_candidates(
  const PlannerInput & input,
  const std::chrono::steady_clock::time_point & deadline) const
{
  if (global_search_enabled_) {
    return explore_global_candidates(input, deadline);
  }
  return explore_local_candidates(input, deadline);
}

std::vector<EndEffectorPlanRelativePoseTask::Candidate>
EndEffectorPlanRelativePoseTask::explore_local_candidates(
  const PlannerInput & input,
  const std::chrono::steady_clock::time_point & deadline) const
{
  std::vector<Candidate> candidates;
  if (input.q_current.size() == 0 || input.relative_jacobian.cols() == 0) {
    return candidates;
  }

  const auto active_indices = active_relative_indices();
  Eigen::MatrixXd active_jacobian(active_indices.size(), input.relative_jacobian.cols());
  const auto error = relative_pose_error(input.current_relative_pose);
  Eigen::VectorXd active_error(active_indices.size());
  for (size_t i = 0; i < active_indices.size(); ++i) {
    active_jacobian.row(static_cast<Eigen::Index>(i)) = input.relative_jacobian.row(active_indices[i]);
    active_error(static_cast<Eigen::Index>(i)) = error(active_indices[i]);
  }

  const Eigen::MatrixXd joint_jacobian =
    active_jacobian.rightCols(static_cast<Eigen::Index>(joint_names_.size()));
  const Eigen::VectorXd projected_goal =
    input.q_current + damped_pseudoinverse(joint_jacobian, 1.0e-6) * active_error;
  const Eigen::MatrixXd nullspace = nullspace_basis(joint_jacobian);

  auto add_candidate = [&](const Eigen::VectorXd & q) {
    if (planner_cancel_.load() || std::chrono::steady_clock::now() >= deadline) {
      return;
    }
    const double error_norm =
      linearized_relative_error_norm(q, input.q_current, joint_jacobian, active_error);
    Candidate candidate = score_candidate(q, input, nullspace, error_norm);
    estimate_local_chart(candidate, nullspace);
    if (validate_candidate(candidate)) {
      candidates.push_back(candidate);
    }
  };

  add_candidate(projected_goal);
  add_candidate(input.q_current);

  const int steps = std::max(1, nullspace_probe_steps_);
  const Eigen::Index max_dirs = std::min<Eigen::Index>(2, nullspace.cols());
  for (Eigen::Index dir = 0; dir < max_dirs; ++dir) {
    for (int step = -steps; step <= steps; ++step) {
      if (step == 0) {
        continue;
      }
      const double alpha = nullspace_probe_radius_ * static_cast<double>(step) / steps;
      add_candidate(projected_goal + alpha * nullspace.col(dir));
    }
  }
  if (max_dirs >= 2) {
    for (int a = -steps; a <= steps; ++a) {
      for (int b = -steps; b <= steps; ++b) {
        if (a == 0 && b == 0) {
          continue;
        }
        const double alpha = nullspace_probe_radius_ * static_cast<double>(a) / steps;
        const double beta = nullspace_probe_radius_ * static_cast<double>(b) / steps;
        add_candidate(projected_goal + alpha * nullspace.col(0) + beta * nullspace.col(1));
      }
    }
  }
  return candidates;
}

std::vector<EndEffectorPlanRelativePoseTask::Candidate>
EndEffectorPlanRelativePoseTask::explore_global_candidates(
  const PlannerInput & input,
  const std::chrono::steady_clock::time_point & deadline) const
{
  std::vector<Candidate> candidates;
  if (input.q_current.size() == 0 || input.relative_jacobian.cols() == 0) {
    return candidates;
  }

  const auto active_indices = active_relative_indices();
  if (active_indices.empty()) {
    return candidates;
  }
  Eigen::MatrixXd active_jacobian(active_indices.size(), input.relative_jacobian.cols());
  const auto error = relative_pose_error(input.current_relative_pose);
  Eigen::VectorXd active_error(active_indices.size());
  for (size_t i = 0; i < active_indices.size(); ++i) {
    active_jacobian.row(static_cast<Eigen::Index>(i)) = input.relative_jacobian.row(active_indices[i]);
    active_error(static_cast<Eigen::Index>(i)) = error(active_indices[i]);
  }
  const Eigen::MatrixXd joint_jacobian =
    active_jacobian.rightCols(static_cast<Eigen::Index>(joint_names_.size()));
  const Eigen::MatrixXd nullspace = nullspace_basis(joint_jacobian);

  AtlasDebugData debug_data;
  const std::size_t seed_count = static_cast<std::size_t>(std::max(1, global_random_starts_));
  auto seeds = generate_sobol_seeds(seed_count);
  debug_data.sobol_seeds = seeds;
  debug_data.projections.resize(seeds.size());
  const std::size_t trace_seed_index = static_cast<std::size_t>(
    std::clamp(global_projection_trace_seed_index_, 0, static_cast<int>(std::max<std::size_t>(1, seeds.size()) - 1)));
  if (global_projection_trace_enabled_ && !seeds.empty()) {
    debug_data.projection_trace.enabled = true;
    debug_data.projection_trace.seed_index = trace_seed_index;
    debug_data.projection_trace.seed = seeds[trace_seed_index];
  }

  const std::size_t workers = std::max<std::size_t>(
    1,
    std::min<std::size_t>(
      seeds.size(),
      static_cast<std::size_t>(std::max(1U, std::thread::hardware_concurrency()))));
  std::vector<std::future<void>> futures;
  futures.reserve(workers);
  for (std::size_t worker = 0; worker < workers; ++worker) {
    futures.push_back(std::async(
        std::launch::async,
        [&, worker]() {
          for (std::size_t index = worker; index < seeds.size(); index += workers) {
            if (planner_cancel_.load() || std::chrono::steady_clock::now() >= deadline) {
              return;
            }
            if (debug_data.projection_trace.enabled && index == trace_seed_index) {
              continue;
            }
            debug_data.projections[index] =
              project_to_manifold(seeds[index], input, joint_jacobian, active_error, deadline);
          }
        }));
  }
  for (auto & future : futures) {
    future.get();
  }
  if (debug_data.projection_trace.enabled) {
    debug_data.projections[trace_seed_index] = project_to_manifold(
      seeds[trace_seed_index],
      input,
      joint_jacobian,
      active_error,
      deadline,
      &debug_data.projection_trace);
  }

  for (std::size_t i = 0; i < debug_data.projections.size(); ++i) {
    const auto & projection = debug_data.projections[i];
    if (!projection.success) {
      std::ostringstream stream;
      stream << "seed S" << i << " rejected: projection_failed error=" << projection.error_norm
             << " iterations=" << projection.iterations
             << " near_explore_threshold=" << projection.near_explore_threshold;
      debug_data.rejection_log.push_back(stream.str());
      continue;
    }
    Candidate candidate = score_candidate(projection.q, input, nullspace, projection.error_norm);
    if (!validate_candidate(candidate)) {
      std::ostringstream stream;
      stream << "seed S" << i << " rejected: candidate_invalid error="
             << candidate.relative_error_norm << " sigma=" << candidate.sigma_min
             << " joint_margin=" << candidate.joint_margin
             << " collision_margin=" << candidate.collision_margin;
      debug_data.rejection_log.push_back(stream.str());
      continue;
    }
    bool duplicate = false;
    for (auto & existing : candidates) {
      if (normalized_distance(existing.q, candidate.q) < global_deduplicate_radius_) {
        duplicate = true;
        const bool candidate_is_better =
          candidate.relative_error_norm < existing.relative_error_norm ||
          (std::abs(candidate.relative_error_norm - existing.relative_error_norm) < 1.0e-9 &&
          candidate.joint_margin > existing.joint_margin);
        if (candidate_is_better) {
          existing = candidate;
        }
        break;
      }
    }
    if (!duplicate) {
      candidates.push_back(candidate);
    }
  }

  auto components = build_atlas_components(
    candidates,
    input,
    joint_jacobian,
    active_error,
    debug_data,
    deadline);
  debug_data.candidates = candidates;
  debug_data.components = components;
  dump_atlas_debug(debug_data, input);

  std::sort(
    candidates.begin(),
    candidates.end(),
    [](const Candidate & lhs, const Candidate & rhs) {
      return lhs.score > rhs.score;
    });

  if (debug_) {
    for (const auto & component : components) {
      std::ostringstream stream;
      stream << "component " << component.id
             << " samples=" << component.indices.size()
             << " score=" << component.score
             << " mean_sigma=" << component.mean_sigma
             << " min_sigma=" << component.min_sigma
             << " mean_manipulability=" << component.mean_manipulability
             << " mean_rel_error=" << component.mean_relative_error
             << " max_rel_error=" << component.max_relative_error
             << " mean_limits=" << component.mean_joint_margin
             << " min_limits=" << component.min_joint_margin
             << " distance=" << component.mean_current_distance
             << " local_area=" << component.local_area_mean
             << " boundary_radius=" << component.boundary_radius_mean
             << " curvature=" << component.curvature_mean;
      debug_log(stream.str());
    }
  }

  return candidates;
}

EndEffectorPlanRelativePoseTask::ProjectionResult
EndEffectorPlanRelativePoseTask::project_to_manifold(
  const Eigen::VectorXd & seed,
  const PlannerInput & input,
  const Eigen::MatrixXd & joint_jacobian,
  const Eigen::VectorXd & active_error,
  const std::chrono::steady_clock::time_point & deadline,
  ProjectionTrace * trace) const
{
  ProjectionResult result;
  result.q = clamp_to_limits(seed, lower_limits_, upper_limits_);
  auto linearization = evaluate_relative_linearization(input, result.q);
  if (!linearization.valid) {
    return result;
  }
  if (trace != nullptr && trace->enabled) {
    trace->clamped_seed = result.q;
    trace->seed_residual = linearization.active_error;
    trace->seed_error_norm = trace->seed_residual.norm();
    trace->explore_seed_threshold = global_explore_threshold_;
  }
  const int max_iterations = std::max(1, global_projection_iterations_);
  const auto active_indices = active_relative_indices();
  for (int iteration = 0; iteration < max_iterations; ++iteration) {
    if (planner_cancel_.load() || std::chrono::steady_clock::now() >= deadline) {
      result.success = false;
      return result;
    }
    linearization = evaluate_relative_linearization(input, result.q);
    if (!linearization.valid || linearization.joint_jacobian.cols() != result.q.size()) {
      result.success = false;
      return result;
    }
    const Eigen::VectorXd residual = linearization.active_error;
    result.error_norm = residual.norm();
    result.iterations = iteration + 1;
    const Eigen::MatrixXd pinv = damped_pseudoinverse(linearization.joint_jacobian, 0.05);
    const Eigen::VectorXd delta_q = pinv * residual;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(linearization.joint_jacobian);
    const Eigen::VectorXd singular_values = svd.singularValues();
    const int rank = numerical_rank(
      singular_values,
      static_cast<int>(linearization.joint_jacobian.rows()),
      static_cast<int>(linearization.joint_jacobian.cols()));
    Eigen::VectorXd full_residual = Eigen::VectorXd::Zero(6);
    for (std::size_t i = 0; i < active_indices.size() && i < static_cast<std::size_t>(residual.size()); ++i) {
      full_residual(active_indices[i]) = residual(static_cast<Eigen::Index>(i));
    }
    const Eigen::VectorXd raw_next = result.q + delta_q;
    const Eigen::VectorXd clamped_next = clamp_to_limits(raw_next, lower_limits_, upper_limits_);
    if (trace != nullptr && trace->enabled) {
      ProjectionIterationTrace iteration_trace;
      iteration_trace.iteration = iteration;
      iteration_trace.q = result.q;
      iteration_trace.residual = full_residual;
      iteration_trace.position_error = full_residual.head<3>();
      iteration_trace.orientation_error = full_residual.tail<3>();
      iteration_trace.delta_q = delta_q;
      iteration_trace.singular_values = singular_values;
      iteration_trace.rank = rank;
      iteration_trace.error_norm = result.error_norm;
      iteration_trace.clamped_joints.resize(static_cast<std::size_t>(raw_next.size()), false);
      for (Eigen::Index joint = 0; joint < raw_next.size(); ++joint) {
        const bool lower_clamped =
          joint < lower_limits_.size() && std::isfinite(lower_limits_(joint)) &&
          raw_next(joint) < lower_limits_(joint);
        const bool upper_clamped =
          joint < upper_limits_.size() && std::isfinite(upper_limits_(joint)) &&
          raw_next(joint) > upper_limits_(joint);
        iteration_trace.clamped_joints[static_cast<std::size_t>(joint)] =
          lower_clamped || upper_clamped || std::abs(clamped_next(joint) - raw_next(joint)) > 1.0e-12;
      }
      trace->iterations.push_back(std::move(iteration_trace));
    }
    if (result.error_norm <= global_projection_tolerance_) {
      result.success = true;
      result.near_explore_threshold = true;
      return result;
    }
    result.q = clamped_next;
  }
  linearization = evaluate_relative_linearization(input, result.q);
  result.error_norm = linearization.valid ?
    linearization.active_error.norm() :
    (active_error - joint_jacobian * (result.q - input.q_current)).norm();
  result.near_explore_threshold = result.error_norm <= global_explore_threshold_;
  result.success = result.error_norm <= global_projection_tolerance_;
  return result;
}

std::vector<EndEffectorPlanRelativePoseTask::Component>
EndEffectorPlanRelativePoseTask::cluster_candidates(
  std::vector<Candidate> & candidates,
  const PlannerInput & input,
  const Eigen::MatrixXd & joint_jacobian,
  const Eigen::VectorXd & active_error,
  const Eigen::MatrixXd & nullspace,
  const std::chrono::steady_clock::time_point & deadline) const
{
  std::vector<Component> components;
  const std::size_t n = candidates.size();
  if (n == 0) {
    return components;
  }

  std::vector<std::vector<std::size_t>> adjacency(n);
  for (std::size_t i = 0; i < n; ++i) {
    std::vector<std::pair<double, std::size_t>> neighbors;
    neighbors.reserve(n > 0 ? n - 1 : 0);
    for (std::size_t j = 0; j < n; ++j) {
      if (i == j) {
        continue;
      }
      const double distance = normalized_distance(candidates[i].q, candidates[j].q);
      if (distance <= global_connect_radius_) {
        neighbors.emplace_back(distance, j);
      }
    }
    std::sort(neighbors.begin(), neighbors.end());
    const std::size_t max_neighbors =
      static_cast<std::size_t>(std::max(0, global_max_edge_neighbors_));
    for (std::size_t k = 0; k < std::min(max_neighbors, neighbors.size()); ++k) {
      const std::size_t j = neighbors[k].second;
      if (i < j && validate_edge(
          candidates[i],
          candidates[j],
          input,
          joint_jacobian,
          active_error,
          nullspace,
          deadline))
      {
        adjacency[i].push_back(j);
        adjacency[j].push_back(i);
      }
    }
  }

  std::vector<int> labels(n, -1);
  int component_id = 0;
  for (std::size_t start = 0; start < n; ++start) {
    if (labels[start] >= 0) {
      continue;
    }
    Component component;
    component.id = component_id;
    std::vector<std::size_t> queue{start};
    labels[start] = component_id;
    for (std::size_t cursor = 0; cursor < queue.size(); ++cursor) {
      const std::size_t node = queue[cursor];
      component.indices.push_back(node);
      for (const auto next : adjacency[node]) {
        if (labels[next] < 0) {
          labels[next] = component_id;
          queue.push_back(next);
        }
      }
    }
    components.push_back(component);
    ++component_id;
  }

  double sigma_scale = 0.0;
  double margin_scale = 0.0;
  double error_scale = global_solution_tolerance_;
  for (const auto & candidate : candidates) {
    sigma_scale = std::max(sigma_scale, candidate.sigma_min);
    margin_scale = std::max(margin_scale, candidate.joint_margin);
    error_scale = std::max(error_scale, candidate.relative_error_norm);
  }
  sigma_scale = std::max(sigma_scale, 1.0e-12);
  margin_scale = std::max(margin_scale, 1.0e-12);
  error_scale = std::max(error_scale, 1.0e-12);
  const double support_scale =
    std::log1p(static_cast<double>(std::max<std::size_t>(1, n)));

  for (auto & component : components) {
    double sigma_sum = 0.0;
    double manipulability_sum = 0.0;
    double error_sum = 0.0;
    double margin_sum = 0.0;
    double distance_sum = 0.0;
    double area_sum = 0.0;
    double radius_sum = 0.0;
    double curvature_sum = 0.0;
    component.min_sigma = std::numeric_limits<double>::infinity();
    component.max_relative_error = 0.0;
    component.min_joint_margin = std::numeric_limits<double>::infinity();

    std::vector<std::size_t> chart_indices = component.indices;
    std::sort(
      chart_indices.begin(),
      chart_indices.end(),
      [&candidates](std::size_t lhs, std::size_t rhs) {
        return candidates[lhs].joint_margin > candidates[rhs].joint_margin;
      });
    if (chart_indices.size() > static_cast<std::size_t>(std::max(1, global_max_charts_per_component_))) {
      chart_indices.resize(static_cast<std::size_t>(std::max(1, global_max_charts_per_component_)));
    }
    for (const auto index : chart_indices) {
      estimate_local_chart(candidates[index], input, joint_jacobian, active_error, nullspace, deadline);
    }

    for (const auto index : component.indices) {
      const auto & candidate = candidates[index];
      sigma_sum += candidate.sigma_min;
      manipulability_sum += candidate.manipulability;
      error_sum += candidate.relative_error_norm;
      margin_sum += candidate.joint_margin;
      distance_sum += candidate.current_distance;
      area_sum += candidate.local_area;
      radius_sum += candidate.boundary_radius;
      curvature_sum += candidate.curvature;
      component.min_sigma = std::min(component.min_sigma, candidate.sigma_min);
      component.max_relative_error = std::max(component.max_relative_error, candidate.relative_error_norm);
      component.min_joint_margin = std::min(component.min_joint_margin, candidate.joint_margin);
    }
    const double count = static_cast<double>(std::max<std::size_t>(1, component.indices.size()));
    component.mean_sigma = sigma_sum / count;
    component.mean_manipulability = manipulability_sum / count;
    component.mean_relative_error = error_sum / count;
    component.mean_joint_margin = margin_sum / count;
    component.mean_current_distance = distance_sum / count;
    component.local_area_mean = area_sum / count;
    component.boundary_radius_mean = radius_sum / count;
    component.curvature_mean = curvature_sum / count;

    double best_representative_score = -std::numeric_limits<double>::infinity();
    for (const auto index : component.indices) {
      const auto & candidate = candidates[index];
      const double representative_score =
        2.0 * normalized_quality(global_solution_tolerance_ - candidate.relative_error_norm, global_solution_tolerance_) +
        candidate.joint_margin -
        0.25 * candidate.current_distance +
        0.1 * candidate.sigma_min;
      if (representative_score > best_representative_score) {
        best_representative_score = representative_score;
        component.representative_index = index;
      }
    }

    const double support =
      support_scale > 0.0 ? std::log1p(count) / support_scale : 1.0;
    const double singleton_penalty = (component.indices.size() <= 1 && n > 1) ? 0.35 : 0.0;
    const double preferred_margin_penalty =
      component.min_joint_margin < preferred_joint_margin_ ?
      (preferred_joint_margin_ - component.min_joint_margin) / std::max(preferred_joint_margin_, 1.0e-9) :
      0.0;
    component.score =
      weight_sigma_ * normalized_quality(component.mean_sigma, sigma_scale) +
      weight_limits_ * normalized_quality(component.mean_joint_margin, margin_scale) +
      weight_room_ * component.local_area_mean +
      weight_boundary_ * component.boundary_radius_mean +
      weight_support_ * support -
      weight_current_distance_ * component.mean_current_distance -
      weight_relative_error_ * normalized_quality(component.max_relative_error, error_scale) -
      weight_curvature_ * component.curvature_mean -
      singleton_penalty -
      preferred_margin_penalty;

    for (const auto index : component.indices) {
      candidates[index].component_id = component.id;
      candidates[index].component_sample_count = static_cast<int>(component.indices.size());
      candidates[index].component_score = component.score;
      candidates[index].score = component.score + 0.15 * candidates[index].joint_margin -
        0.05 * candidates[index].current_distance -
        weight_relative_error_ * normalized_quality(candidates[index].relative_error_norm, error_scale);
    }
    candidates[component.representative_index].score += 0.05;
  }

  std::sort(
    components.begin(),
    components.end(),
    [](const Component & lhs, const Component & rhs) {
      return lhs.score > rhs.score;
    });
  return components;
}

std::vector<EndEffectorPlanRelativePoseTask::Component>
EndEffectorPlanRelativePoseTask::build_atlas_components(
  std::vector<Candidate> & candidates,
  const PlannerInput & input,
  const Eigen::MatrixXd & joint_jacobian,
  const Eigen::VectorXd & active_error,
  AtlasDebugData & debug_data,
  const std::chrono::steady_clock::time_point & deadline) const
{
  std::vector<Component> components;
  if (candidates.empty()) {
    return components;
  }

  std::vector<std::vector<std::size_t>> adjacency(candidates.size());
  std::deque<std::size_t> queue;
  const int max_seed_charts = std::max(1, global_atlas_max_charts_per_seed_);
  const std::size_t max_total_charts =
    static_cast<std::size_t>(std::max(1, global_atlas_max_total_charts_));
  const double step = std::max(1.0e-6, global_atlas_step_);

  auto candidate_nullspace = [&](const Candidate & candidate) {
      const auto linearization = evaluate_relative_linearization(input, candidate.q);
      if (linearization.valid) {
        return nullspace_basis(linearization.joint_jacobian);
      }
      return nullspace_basis(joint_jacobian);
    };

  auto connect_candidates = [&](std::size_t a_index, std::size_t b_index, const std::string & reason) {
      if (a_index == b_index || a_index >= candidates.size() || b_index >= candidates.size()) {
        return false;
      }
      for (const auto existing : adjacency[a_index]) {
        if (existing == b_index) {
          return true;
        }
      }
      const Eigen::MatrixXd ns = candidate_nullspace(candidates[a_index]);
      if (!validate_edge(candidates[a_index], candidates[b_index], input, joint_jacobian, active_error, ns, deadline)) {
        return false;
      }
      adjacency[a_index].push_back(b_index);
      adjacency[b_index].push_back(a_index);
      AtlasEdge edge;
      edge.from = a_index;
      edge.to = b_index;
      edge.distance = normalized_distance(candidates[a_index].q, candidates[b_index].q);
      edge.reason = reason;
      debug_data.edges.push_back(edge);
      return true;
    };

  auto find_duplicate = [&](const Candidate & candidate) -> std::pair<bool, std::size_t> {
      for (std::size_t i = 0; i < candidates.size(); ++i) {
        if (normalized_distance(candidates[i].q, candidate.q) < global_deduplicate_radius_) {
          return {true, i};
        }
      }
      return {false, 0};
    };

  auto add_chart = [&](
      std::size_t candidate_index,
      std::size_t root_candidate_index,
      std::size_t parent_chart_id,
      int depth) {
      if (debug_data.charts.size() >= max_total_charts || candidate_index >= candidates.size()) {
        return false;
      }
      AtlasChart chart;
      chart.id = debug_data.charts.size();
      chart.candidate_index = candidate_index;
      chart.root_candidate_index = root_candidate_index;
      chart.parent_chart_id = parent_chart_id;
      chart.depth = depth;
      chart.tangent_basis = candidate_nullspace(candidates[candidate_index]);
      estimate_local_chart(
        candidates[candidate_index],
        input,
        joint_jacobian,
        active_error,
        chart.tangent_basis,
        deadline);
      chart.chart_radius = candidates[candidate_index].boundary_radius;
      chart.local_area = candidates[candidate_index].local_area;
      chart.boundary_radius = candidates[candidate_index].boundary_radius;
      chart.curvature = candidates[candidate_index].curvature;
      debug_data.charts.push_back(chart);
      queue.push_back(chart.id);
      return true;
    };

  const std::size_t initial_candidate_count = candidates.size();
  for (std::size_t i = 0; i < initial_candidate_count; ++i) {
    add_chart(i, i, debug_data.charts.size(), 0);
  }

  std::vector<int> charts_from_root(initial_candidate_count, 1);
  while (!queue.empty()) {
    if (planner_cancel_.load() || std::chrono::steady_clock::now() >= deadline ||
      debug_data.charts.size() >= max_total_charts)
    {
      break;
    }
    const std::size_t chart_id = queue.front();
    queue.pop_front();
    if (chart_id >= debug_data.charts.size()) {
      continue;
    }
    if (debug_data.charts[chart_id].expanded ||
      debug_data.charts[chart_id].candidate_index >= candidates.size())
    {
      continue;
    }
    debug_data.charts[chart_id].expanded = true;
    const AtlasChart chart = debug_data.charts[chart_id];
    const Eigen::MatrixXd basis = chart.tangent_basis;
    const Eigen::Index dims = std::min<Eigen::Index>(2, basis.cols());
    if (dims <= 0) {
      debug_data.rejection_log.push_back("chart C" + std::to_string(chart_id) + " not expanded: empty_nullspace");
      continue;
    }

    std::vector<Eigen::VectorXd> directions;
    for (Eigen::Index dim = 0; dim < dims; ++dim) {
      directions.push_back(basis.col(dim));
      directions.push_back(-basis.col(dim));
    }
    if (dims == 2) {
      for (const double sx : {-1.0, 1.0}) {
        for (const double sy : {-1.0, 1.0}) {
          directions.push_back((sx * basis.col(0) + sy * basis.col(1)).normalized());
        }
      }
    }

    for (const auto & direction : directions) {
      if (planner_cancel_.load() || std::chrono::steady_clock::now() >= deadline ||
        debug_data.charts.size() >= max_total_charts)
      {
        break;
      }
      const Eigen::VectorXd seed = candidates[chart.candidate_index].q + step * direction;
      const auto projection = project_to_manifold(seed, input, joint_jacobian, active_error, deadline);
      if (!projection.success || projection.error_norm > global_solution_tolerance_) {
        debug_data.rejection_log.push_back(
          "chart C" + std::to_string(chart_id) + " step rejected: projection_failed");
        continue;
      }
      if (normalized_distance(projection.q, seed) > global_max_edge_projection_distance_) {
        debug_data.rejection_log.push_back(
          "chart C" + std::to_string(chart_id) + " step rejected: large_corrector_distance");
        continue;
      }
      Candidate next = score_candidate(projection.q, input, basis, projection.error_norm);
      if (!validate_candidate(next)) {
        debug_data.rejection_log.push_back(
          "chart C" + std::to_string(chart_id) + " step rejected: invalid_candidate");
        continue;
      }

      const auto duplicate = find_duplicate(next);
      std::size_t next_index = duplicate.second;
      if (!duplicate.first) {
        next_index = candidates.size();
        candidates.push_back(next);
        adjacency.emplace_back();
      }

      connect_candidates(chart.candidate_index, next_index, "predictor_corrector");

      if (!duplicate.first) {
        const std::size_t root = chart.root_candidate_index;
        if (root < charts_from_root.size() && charts_from_root[root] < max_seed_charts) {
          ++charts_from_root[root];
          add_chart(next_index, root, chart_id, chart.depth + 1);
        }
      }

      for (std::size_t other = 0; other < candidates.size(); ++other) {
        if (other == next_index) {
          continue;
        }
        if (normalized_distance(candidates[other].q, candidates[next_index].q) <= global_atlas_neighbor_radius_) {
          connect_candidates(next_index, other, "verified_neighbor_transition");
        }
      }
    }
  }

  std::vector<int> labels(candidates.size(), -1);
  int component_id = 0;
  for (std::size_t start = 0; start < candidates.size(); ++start) {
    if (labels[start] >= 0) {
      continue;
    }
    Component component;
    component.id = component_id;
    std::deque<std::size_t> bfs;
    bfs.push_back(start);
    labels[start] = component_id;
    while (!bfs.empty()) {
      const std::size_t node = bfs.front();
      bfs.pop_front();
      component.indices.push_back(node);
      for (const auto next : adjacency[node]) {
        if (labels[next] < 0) {
          labels[next] = component_id;
          bfs.push_back(next);
        }
      }
    }
    components.push_back(component);
    ++component_id;
  }

  double sigma_scale = 0.0;
  double margin_scale = 0.0;
  double error_scale = global_solution_tolerance_;
  for (const auto & candidate : candidates) {
    sigma_scale = std::max(sigma_scale, candidate.sigma_min);
    margin_scale = std::max(margin_scale, candidate.joint_margin);
    error_scale = std::max(error_scale, candidate.relative_error_norm);
  }
  sigma_scale = std::max(sigma_scale, 1.0e-12);
  margin_scale = std::max(margin_scale, 1.0e-12);
  error_scale = std::max(error_scale, 1.0e-12);
  const double support_scale =
    std::log1p(static_cast<double>(std::max<std::size_t>(1, candidates.size())));

  for (auto & component : components) {
    double sigma_sum = 0.0;
    double manipulability_sum = 0.0;
    double error_sum = 0.0;
    double margin_sum = 0.0;
    double distance_sum = 0.0;
    double area_sum = 0.0;
    double radius_sum = 0.0;
    double curvature_sum = 0.0;
    component.min_sigma = std::numeric_limits<double>::infinity();
    component.max_relative_error = 0.0;
    component.min_joint_margin = std::numeric_limits<double>::infinity();

    double best_representative_score = -std::numeric_limits<double>::infinity();
    for (const auto index : component.indices) {
      auto & candidate = candidates[index];
      if (candidate.local_area <= 0.0 && candidate.boundary_radius <= 0.0) {
        const Eigen::MatrixXd ns = candidate_nullspace(candidate);
        estimate_local_chart(candidate, input, joint_jacobian, active_error, ns, deadline);
      }
      sigma_sum += candidate.sigma_min;
      manipulability_sum += candidate.manipulability;
      error_sum += candidate.relative_error_norm;
      margin_sum += candidate.joint_margin;
      distance_sum += candidate.current_distance;
      area_sum += candidate.local_area;
      radius_sum += candidate.boundary_radius;
      curvature_sum += candidate.curvature;
      component.min_sigma = std::min(component.min_sigma, candidate.sigma_min);
      component.max_relative_error = std::max(component.max_relative_error, candidate.relative_error_norm);
      component.min_joint_margin = std::min(component.min_joint_margin, candidate.joint_margin);

      const double representative_score =
        2.0 * normalized_quality(global_solution_tolerance_ - candidate.relative_error_norm, global_solution_tolerance_) +
        candidate.joint_margin -
        0.25 * candidate.current_distance +
        0.1 * candidate.sigma_min +
        0.05 * candidate.local_area;
      if (representative_score > best_representative_score) {
        best_representative_score = representative_score;
        component.representative_index = index;
      }
    }

    const double count = static_cast<double>(std::max<std::size_t>(1, component.indices.size()));
    component.mean_sigma = sigma_sum / count;
    component.mean_manipulability = manipulability_sum / count;
    component.mean_relative_error = error_sum / count;
    component.mean_joint_margin = margin_sum / count;
    component.mean_current_distance = distance_sum / count;
    component.local_area_mean = area_sum / count;
    component.boundary_radius_mean = radius_sum / count;
    component.curvature_mean = curvature_sum / count;

    const double support = support_scale > 0.0 ? std::log1p(count) / support_scale : 1.0;
    const double singleton_penalty = (component.indices.size() <= 1 && candidates.size() > 1) ? 0.35 : 0.0;
    const double preferred_margin_penalty =
      component.min_joint_margin < preferred_joint_margin_ ?
      (preferred_joint_margin_ - component.min_joint_margin) / std::max(preferred_joint_margin_, 1.0e-9) :
      0.0;
    component.score =
      weight_sigma_ * normalized_quality(component.mean_sigma, sigma_scale) +
      weight_limits_ * normalized_quality(component.mean_joint_margin, margin_scale) +
      weight_room_ * component.local_area_mean +
      weight_boundary_ * component.boundary_radius_mean +
      weight_support_ * support -
      weight_current_distance_ * component.mean_current_distance -
      weight_relative_error_ * normalized_quality(component.max_relative_error, error_scale) -
      weight_curvature_ * component.curvature_mean -
      singleton_penalty -
      preferred_margin_penalty;

    for (const auto index : component.indices) {
      candidates[index].component_id = component.id;
      candidates[index].component_sample_count = static_cast<int>(component.indices.size());
      candidates[index].component_score = component.score;
      candidates[index].score = component.score + 0.15 * candidates[index].joint_margin -
        0.05 * candidates[index].current_distance -
        weight_relative_error_ * normalized_quality(candidates[index].relative_error_norm, error_scale);
    }
    candidates[component.representative_index].score += 0.05;
  }

  for (auto & chart : debug_data.charts) {
    if (chart.candidate_index < candidates.size()) {
      chart.component_id = candidates[chart.candidate_index].component_id;
    }
  }

  std::sort(
    components.begin(),
    components.end(),
    [](const Component & lhs, const Component & rhs) {
      return lhs.score > rhs.score;
    });
  return components;
}

bool EndEffectorPlanRelativePoseTask::validate_edge(
  const Candidate & a,
  const Candidate & b,
  const PlannerInput & input,
  const Eigen::MatrixXd & joint_jacobian,
  const Eigen::VectorXd & active_error,
  const Eigen::MatrixXd & nullspace,
  const std::chrono::steady_clock::time_point & deadline) const
{
  const int steps = std::max(2, global_path_check_steps_);
  for (int step = 0; step <= steps; ++step) {
    const double alpha = static_cast<double>(step) / static_cast<double>(steps);
    const Eigen::VectorXd seed = a.q + alpha * (b.q - a.q);
    const auto projection = project_to_manifold(seed, input, joint_jacobian, active_error, deadline);
    if (!projection.success) {
      return false;
    }
    if (projection.error_norm > global_solution_tolerance_) {
      return false;
    }
    if (normalized_distance(projection.q, seed) > global_max_edge_projection_distance_) {
      return false;
    }
    const Candidate candidate = score_candidate(projection.q, input, nullspace, projection.error_norm);
    if (!validate_candidate(candidate)) {
      return false;
    }
  }
  return true;
}

void EndEffectorPlanRelativePoseTask::estimate_local_chart(
  Candidate & candidate,
  const Eigen::MatrixXd & nullspace) const
{
  candidate.local_area = 0.0;
  candidate.boundary_radius = 0.0;
  if (nullspace.cols() == 0 || !within_limits(candidate.q)) {
    return;
  }
  const Eigen::Index dims = std::min<Eigen::Index>(2, nullspace.cols());
  Eigen::MatrixXd basis = nullspace.leftCols(dims);
  if (dims == 1) {
    int accepted = 0;
    double radius_sum = 0.0;
    for (const double sign : {-1.0, 1.0}) {
      double last_radius = 0.0;
      for (double radius = global_polar_radial_step_;
        radius <= global_polar_max_radius_ + 0.5 * global_polar_radial_step_;
        radius += global_polar_radial_step_)
      {
        const Eigen::VectorXd q = candidate.q + sign * radius * basis.col(0);
        if (!within_limits(q) || joint_margin(q) < min_joint_margin_) {
          break;
        }
        last_radius = radius;
      }
      radius_sum += last_radius;
      if (last_radius > 0.0) {
        ++accepted;
      }
    }
    candidate.boundary_radius = accepted > 0 ? radius_sum / static_cast<double>(accepted) : 0.0;
    candidate.local_area = 2.0 * candidate.boundary_radius;
    return;
  }

  std::vector<double> radii;
  const int angular_samples = std::max(1, global_polar_angular_samples_);
  radii.reserve(static_cast<std::size_t>(angular_samples));
  constexpr double kPi = 3.14159265358979323846;
  for (int sample = 0; sample < angular_samples; ++sample) {
    const double theta =
      2.0 * kPi * static_cast<double>(sample) / static_cast<double>(angular_samples);
    const Eigen::VectorXd direction =
      std::cos(theta) * basis.col(0) + std::sin(theta) * basis.col(1);
    double last_radius = 0.0;
    for (double radius = global_polar_radial_step_;
      radius <= global_polar_max_radius_ + 0.5 * global_polar_radial_step_;
      radius += global_polar_radial_step_)
    {
      const Eigen::VectorXd q = candidate.q + radius * direction;
      if (!within_limits(q) || joint_margin(q) < min_joint_margin_) {
        break;
      }
      last_radius = radius;
    }
    radii.push_back(last_radius);
  }
  double squared_sum = 0.0;
  double radius_sum = 0.0;
  for (const double radius : radii) {
    squared_sum += radius * radius;
    radius_sum += radius;
  }
  const double delta_theta = 2.0 * kPi / static_cast<double>(angular_samples);
  candidate.local_area = 0.5 * squared_sum * delta_theta;
  candidate.boundary_radius = radii.empty() ? 0.0 : radius_sum / static_cast<double>(radii.size());
}

void EndEffectorPlanRelativePoseTask::estimate_local_chart(
  Candidate & candidate,
  const PlannerInput & input,
  const Eigen::MatrixXd & joint_jacobian,
  const Eigen::VectorXd & active_error,
  const Eigen::MatrixXd & nullspace,
  const std::chrono::steady_clock::time_point & deadline) const
{
  candidate.local_area = 0.0;
  candidate.boundary_radius = 0.0;
  candidate.curvature = 0.0;
  if (nullspace.cols() == 0 || !within_limits(candidate.q)) {
    return;
  }

  const Eigen::Index dims = std::min<Eigen::Index>(2, nullspace.cols());
  const Eigen::MatrixXd basis = nullspace.leftCols(dims);
  auto probe = [&](const Eigen::VectorXd & seed, double radius) {
      const auto projection = project_to_manifold(seed, input, joint_jacobian, active_error, deadline);
      if (!projection.success || projection.error_norm > global_solution_tolerance_) {
        return false;
      }
      if (normalized_distance(projection.q, seed) > global_max_edge_projection_distance_) {
        return false;
      }
      const Candidate projected = score_candidate(projection.q, input, nullspace, projection.error_norm);
      if (!validate_candidate(projected)) {
        return false;
      }
      const double projector_change =
        radius > 1.0e-12 ? std::abs(projected.nullspace_room - candidate.nullspace_room) / radius : 0.0;
      return projector_change <= global_max_projector_change_;
    };

  if (dims == 1) {
    int accepted = 0;
    double radius_sum = 0.0;
    for (const double sign : {-1.0, 1.0}) {
      double last_radius = 0.0;
      for (double radius = global_polar_radial_step_;
        radius <= global_polar_max_radius_ + 0.5 * global_polar_radial_step_;
        radius += global_polar_radial_step_)
      {
        const Eigen::VectorXd seed = candidate.q + sign * radius * basis.col(0);
        if (!probe(seed, radius)) {
          break;
        }
        last_radius = radius;
      }
      radius_sum += last_radius;
      if (last_radius > 0.0) {
        ++accepted;
      }
    }
    candidate.boundary_radius = accepted > 0 ? radius_sum / static_cast<double>(accepted) : 0.0;
    candidate.local_area = 2.0 * candidate.boundary_radius;
    return;
  }

  std::vector<double> radii;
  const int angular_samples = std::max(1, global_polar_angular_samples_);
  radii.reserve(static_cast<std::size_t>(angular_samples));
  constexpr double kPi = 3.14159265358979323846;
  for (int sample = 0; sample < angular_samples; ++sample) {
    const double theta =
      2.0 * kPi * static_cast<double>(sample) / static_cast<double>(angular_samples);
    const Eigen::VectorXd direction =
      std::cos(theta) * basis.col(0) + std::sin(theta) * basis.col(1);
    double last_radius = 0.0;
    for (double radius = global_polar_radial_step_;
      radius <= global_polar_max_radius_ + 0.5 * global_polar_radial_step_;
      radius += global_polar_radial_step_)
    {
      const Eigen::VectorXd seed = candidate.q + radius * direction;
      if (!probe(seed, radius)) {
        break;
      }
      last_radius = radius;
    }
    radii.push_back(last_radius);
  }

  double squared_sum = 0.0;
  double radius_sum = 0.0;
  for (const double radius : radii) {
    squared_sum += radius * radius;
    radius_sum += radius;
  }
  const double delta_theta = 2.0 * kPi / static_cast<double>(angular_samples);
  candidate.local_area = 0.5 * squared_sum * delta_theta;
  candidate.boundary_radius = radii.empty() ? 0.0 : radius_sum / static_cast<double>(radii.size());
}

bool EndEffectorPlanRelativePoseTask::validate_candidate(const Candidate & candidate) const
{
  const double tolerance = global_search_enabled_ ?
    std::min(global_projection_tolerance_, global_solution_tolerance_) : reprojection_tolerance_;
  return within_limits(candidate.q) &&
    candidate.relative_error_norm <= tolerance &&
    candidate.sigma_min >= min_sigma_ &&
    candidate.joint_margin >= min_joint_margin_ &&
    candidate.collision_margin >= min_collision_margin_;
}

bool EndEffectorPlanRelativePoseTask::validate_trajectory(
  const Eigen::VectorXd & start,
  const Eigen::VectorXd & goal) const
{
  const int steps = std::max(2, nullspace_probe_steps_ * 2);
  for (int i = 0; i <= steps; ++i) {
    const double alpha = static_cast<double>(i) / static_cast<double>(steps);
    const Eigen::VectorXd q = start + alpha * (goal - start);
    if (!within_limits(q) || joint_margin(q) < min_joint_margin_) {
      return false;
    }
  }
  return true;
}

bool EndEffectorPlanRelativePoseTask::request_external_trajectory(
  const Eigen::VectorXd & start,
  const Eigen::VectorXd & goal,
  const std::chrono::steady_clock::time_point & deadline,
  PlannedTrajectory & trajectory,
  std::string & message,
  bool & timed_out)
{
  timed_out = false;
  if (!external_planner_client_) {
    message = "external_planner_client_unavailable";
    return false;
  }

  const auto now = std::chrono::steady_clock::now();
  if (now >= deadline) {
    timed_out = true;
    message = "planning_timeout";
    return false;
  }
  const double remaining = std::chrono::duration<double>(deadline - now).count();
  const double wait_timeout = std::min(external_planner_wait_timeout_, remaining);
  debug_log("waiting for external bimanual planner action server");
  if (!external_planner_client_->wait_for_action_server(std::chrono::duration<double>(wait_timeout))) {
    message = "external_planner_unavailable";
    return false;
  }

  PlanBimanualJointTrajectory::Goal action_goal;
  action_goal.joint_names = joint_names_;
  action_goal.joint_positions.reserve(static_cast<std::size_t>(goal.size()));
  for (Eigen::Index i = 0; i < goal.size(); ++i) {
    action_goal.joint_positions.push_back(goal(i));
  }
  const auto after_wait = std::chrono::steady_clock::now();
  if (after_wait >= deadline) {
    timed_out = true;
    message = "planning_timeout";
    return false;
  }
  action_goal.planning_time = std::chrono::duration<double>(deadline - after_wait).count();
  action_goal.planning_time = std::min(action_goal.planning_time, external_planner_result_timeout_);
  action_goal.goal_tolerance = external_planner_goal_tolerance_;
  action_goal.execute = false;

  debug_log("sending selected manifold to external bimanual planner");
  auto goal_handle_future = external_planner_client_->async_send_goal(action_goal);
  while (rclcpp::ok()) {
    if (planner_cancel_.load()) {
      message = "planning_canceled";
      return false;
    }
    if (std::chrono::steady_clock::now() >= deadline) {
      timed_out = true;
      message = "planning_timeout";
      return false;
    }
    if (goal_handle_future.wait_for(std::chrono::milliseconds(10)) == std::future_status::ready) {
      break;
    }
  }

  const auto goal_handle = goal_handle_future.get();
  if (!goal_handle) {
    message = "external_planner_rejected_goal";
    return false;
  }

  auto result_future = external_planner_client_->async_get_result(goal_handle);
  const auto result_deadline = std::min(
    deadline,
    std::chrono::steady_clock::now() +
    std::chrono::duration_cast<std::chrono::steady_clock::duration>(
      std::chrono::duration<double>(external_planner_result_timeout_)));
  while (rclcpp::ok()) {
    if (planner_cancel_.load()) {
      external_planner_client_->async_cancel_goal(goal_handle);
      message = "planning_canceled";
      return false;
    }
    if (std::chrono::steady_clock::now() >= result_deadline) {
      external_planner_client_->async_cancel_goal(goal_handle);
      timed_out = true;
      message = "planning_timeout";
      return false;
    }
    if (result_future.wait_for(std::chrono::milliseconds(10)) == std::future_status::ready) {
      break;
    }
  }

  const auto wrapped_result = result_future.get();
  if (wrapped_result.code != rclcpp_action::ResultCode::SUCCEEDED || !wrapped_result.result) {
    message = "external_planner_failed";
    if (wrapped_result.result && !wrapped_result.result->message.empty()) {
      message += ": " + wrapped_result.result->message;
    }
    return false;
  }
  if (!wrapped_result.result->success) {
    message = wrapped_result.result->message.empty() ?
      "external_planner_failed" : wrapped_result.result->message;
    return false;
  }
  return trajectory_from_msg(
    wrapped_result.result->trajectory,
    start,
    goal,
    trajectory,
    message);
}

bool EndEffectorPlanRelativePoseTask::trajectory_from_msg(
  const trajectory_msgs::msg::JointTrajectory & msg,
  const Eigen::VectorXd & start,
  const Eigen::VectorXd & goal,
  PlannedTrajectory & trajectory,
  std::string & message) const
{
  if (msg.points.empty()) {
    message = "external_planner_returned_empty_trajectory";
    return false;
  }
  std::vector<std::size_t> source_indices;
  source_indices.reserve(joint_names_.size());
  for (const auto & joint_name : joint_names_) {
    const auto it = std::find(msg.joint_names.begin(), msg.joint_names.end(), joint_name);
    if (it == msg.joint_names.end()) {
      message = "external_trajectory_missing_joint_" + joint_name;
      return false;
    }
    source_indices.push_back(static_cast<std::size_t>(std::distance(msg.joint_names.begin(), it)));
  }

  PlannedTrajectory converted;
  converted.start = start;
  converted.goal = goal;
  double previous_time = -std::numeric_limits<double>::infinity();
  for (const auto & point : msg.points) {
    Eigen::VectorXd q = Eigen::VectorXd::Zero(joint_names_.size());
    for (std::size_t i = 0; i < source_indices.size(); ++i) {
      const auto source_index = source_indices[i];
      if (source_index >= point.positions.size()) {
        message = "external_trajectory_point_missing_positions";
        return false;
      }
      q(static_cast<Eigen::Index>(i)) = point.positions[source_index];
    }
    const double time = duration_to_seconds(point.time_from_start);
    if (!std::isfinite(time) || time < previous_time) {
      message = "external_trajectory_has_invalid_timing";
      return false;
    }
    if (!within_limits(q)) {
      message = "external_trajectory_violates_joint_limits";
      return false;
    }
    converted.points.push_back(q);
    converted.times.push_back(time);
    previous_time = time;
  }
  converted.duration = converted.times.empty() ? 0.0 : converted.times.back();
  if (converted.duration <= 0.0) {
    converted.duration =
      std::max(0.1, (goal - start).cwiseAbs().maxCoeff() / std::max(1.0e-6, max_joint_velocity_));
  }
  trajectory = std::move(converted);
  return true;
}

EndEffectorPlanRelativePoseTask::Candidate EndEffectorPlanRelativePoseTask::score_candidate(
  const Eigen::VectorXd & q,
  const PlannerInput & input,
  const Eigen::MatrixXd & nullspace,
  double relative_error_norm) const
{
  const auto linearization = evaluate_relative_linearization(input, q);
  const Eigen::MatrixXd & candidate_jacobian =
    linearization.valid ? linearization.relative_jacobian : input.relative_jacobian;
  const Eigen::MatrixXd candidate_nullspace =
    linearization.valid ? nullspace_basis(linearization.joint_jacobian) : nullspace;
  const double candidate_error =
    linearization.valid ? linearization.active_error.norm() : relative_error_norm;
  return score_candidate(q, input, candidate_nullspace, candidate_jacobian, candidate_error);
}

EndEffectorPlanRelativePoseTask::Candidate EndEffectorPlanRelativePoseTask::score_candidate(
  const Eigen::VectorXd & q,
  const PlannerInput & input,
  const Eigen::MatrixXd & nullspace,
  const Eigen::MatrixXd & relative_jacobian,
  double relative_error_norm) const
{
  Candidate candidate;
  candidate.q = q;
  candidate.relative_error_norm = relative_error_norm;
  candidate.sigma_min = smallest_singular_value(relative_jacobian);
  candidate.manipulability = manipulability_measure(relative_jacobian);
  candidate.joint_margin = joint_margin(q);
  candidate.collision_margin = std::numeric_limits<double>::infinity();
  candidate.nullspace_room = 0.0;
  if (nullspace.cols() > 0) {
    for (Eigen::Index i = 0; i < nullspace.cols(); ++i) {
      const Eigen::VectorXd plus = q + nullspace_probe_radius_ * nullspace.col(i);
      const Eigen::VectorXd minus = q - nullspace_probe_radius_ * nullspace.col(i);
      if (within_limits(plus)) {
        candidate.nullspace_room += nullspace_probe_radius_;
      }
      if (within_limits(minus)) {
        candidate.nullspace_room += nullspace_probe_radius_;
      }
    }
  }
  candidate.smoothness = 1.0 / (1.0 + normalized_distance(q, input.q_current));
  candidate.current_distance = normalized_distance(q, input.q_current);
  candidate.curvature = 0.0;
  const double collision_score = std::isfinite(candidate.collision_margin) ?
    candidate.collision_margin : min_collision_margin_;
  const double preferred_margin_penalty =
    candidate.joint_margin < preferred_joint_margin_ ?
    (preferred_joint_margin_ - candidate.joint_margin) / std::max(preferred_joint_margin_, 1.0e-9) :
    0.0;
  candidate.score =
    weight_sigma_ * candidate.sigma_min +
    weight_collision_ * collision_score +
    weight_limits_ * candidate.joint_margin +
    weight_room_ * candidate.nullspace_room +
    weight_smoothness_ * candidate.smoothness -
    weight_current_distance_ * candidate.current_distance -
    weight_curvature_ * candidate.curvature -
    preferred_margin_penalty;
  return candidate;
}

TaskComputation EndEffectorPlanRelativePoseTask::move_to_manifold_update(
  const WholeBodyState & state,
  bool allow_transition)
{
  TaskComputation computation;
  computation.active = false;
  if (!state.joints_valid || state.joint_positions.size() != static_cast<Eigen::Index>(joint_names_.size())) {
    computation.status_message = "waiting_for_joints";
    return computation;
  }

  const auto now = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(now - trajectory_start_).count();
  const Eigen::VectorXd target = sample_trajectory(elapsed);
  current_joint_target_ = target;

  computation.active = true;
  computation.jacobian = Eigen::MatrixXd::Zero(joint_names_.size(), context_.model->total_dofs());
  computation.error = target - state.joint_positions;
  computation.desired_velocity = Eigen::VectorXd::Zero(joint_names_.size());
  for (size_t row = 0; row < joint_names_.size(); ++row) {
    const auto r = static_cast<Eigen::Index>(row);
    computation.jacobian(r, static_cast<Eigen::Index>(context_.model->base_dofs() + row)) = 1.0;
    computation.desired_velocity(r) = std::clamp(
      move_gain_ * computation.error(r),
      -max_joint_velocity_,
      max_joint_velocity_);
  }

  if (allow_transition && elapsed >= active_trajectory_.duration &&
    computation.error.norm() <= move_goal_tolerance_)
  {
    std::lock_guard<std::mutex> lock(mutex_);
    mode_ = Mode::kBlendToRelativePose;
    blend_start_ = now;
    status_message_ = "blending_to_relative_pose";
  } else {
    computation.status_message = "moving_to_manifold";
  }
  return computation;
}

TaskComputation EndEffectorPlanRelativePoseTask::blend_update(
  const WholeBodyState & state,
  const KinematicsBackend & backend)
{
  const auto now = std::chrono::steady_clock::now();
  const double elapsed = std::chrono::duration<double>(now - blend_start_).count();
  const double relative_weight = blend_duration_ <= 0.0 ?
    1.0 : std::clamp(elapsed / blend_duration_, 0.0, 1.0);
  const double joint_weight = 1.0 - relative_weight;

  TaskComputation joint_task = move_to_manifold_update(state, false);
  TaskComputation relative_task = relative_pose_update(backend, relative_weight);
  if (relative_weight >= 1.0) {
    std::lock_guard<std::mutex> lock(mutex_);
    mode_ = Mode::kServoRelativePose;
    status_message_ = "servo_relative_pose";
    return relative_pose_update(backend, 1.0);
  }
  if (!joint_task.active) {
    return relative_task;
  }

  joint_task.jacobian *= joint_weight;
  joint_task.desired_velocity *= joint_weight;
  joint_task.error *= joint_weight;
  if (!relative_task.active) {
    joint_task.status_message = "blending_to_relative_pose";
    return joint_task;
  }

  TaskComputation out;
  out.active = true;
  out.jacobian = Eigen::MatrixXd::Zero(
    joint_task.jacobian.rows() + relative_task.jacobian.rows(),
    joint_task.jacobian.cols());
  out.jacobian.topRows(joint_task.jacobian.rows()) = joint_task.jacobian;
  out.jacobian.bottomRows(relative_task.jacobian.rows()) = relative_task.jacobian;
  out.desired_velocity = Eigen::VectorXd::Zero(
    joint_task.desired_velocity.size() + relative_task.desired_velocity.size());
  out.desired_velocity.head(joint_task.desired_velocity.size()) = joint_task.desired_velocity;
  out.desired_velocity.tail(relative_task.desired_velocity.size()) = relative_task.desired_velocity;
  out.error = Eigen::VectorXd::Zero(joint_task.error.size() + relative_task.error.size());
  out.error.head(joint_task.error.size()) = joint_task.error;
  out.error.tail(relative_task.error.size()) = relative_task.error;
  out.has_frame_pose = relative_task.has_frame_pose;
  out.frame_id = relative_task.frame_id;
  out.frame_pose = relative_task.frame_pose;
  out.status_message = "blending_to_relative_pose";
  return out;
}

TaskComputation EndEffectorPlanRelativePoseTask::relative_pose_update(
  const KinematicsBackend & backend,
  double row_scale) const
{
  TaskComputation computation;
  computation.active = enabled_;
  if (!enabled_) {
    computation.status_message = "disabled";
    return computation;
  }
  if (!has_target_relative_pose_) {
    computation.active = false;
    computation.status_message = "waiting_for_target";
    return computation;
  }

  const FrameState reference = backend.get_frame_state(reference_frame_);
  const FrameState controlled = backend.get_frame_state(controlled_frame_);
  const Eigen::Isometry3d current_relative_pose = reference.pose.inverse() * controlled.pose;
  const Eigen::Matrix<double, 6, 1> full_error = relative_pose_error(current_relative_pose);
  const Eigen::MatrixXd full_jacobian = relative_jacobian(reference, controlled);
  const auto active_indices = active_relative_indices();

  computation.has_frame_pose = true;
  computation.frame_id = controlled_frame_;
  computation.frame_pose = current_relative_pose;
  if (active_indices.empty() || row_scale <= 0.0) {
    computation.active = false;
    computation.status_message = "no active axes";
    return computation;
  }

  computation.error = Eigen::VectorXd::Zero(active_indices.size());
  computation.desired_velocity = Eigen::VectorXd::Zero(active_indices.size());
  computation.jacobian = Eigen::MatrixXd::Zero(
    static_cast<Eigen::Index>(active_indices.size()), full_jacobian.cols());
  for (size_t row = 0; row < active_indices.size(); ++row) {
    const Eigen::Index source_row = active_indices[row];
    computation.error(static_cast<Eigen::Index>(row)) = row_scale * full_error(source_row);
    computation.desired_velocity(static_cast<Eigen::Index>(row)) =
      row_scale * gains_(source_row) * full_error(source_row);
    computation.jacobian.row(static_cast<Eigen::Index>(row)) =
      row_scale * full_jacobian.row(source_row);
  }
  computation.status_message = "tracking_relative_pose";
  return computation;
}

Eigen::MatrixXd EndEffectorPlanRelativePoseTask::relative_jacobian(
  const FrameState & reference,
  const FrameState & controlled) const
{
  Eigen::MatrixXd full_jacobian = Eigen::MatrixXd::Zero(6, reference.jacobian.cols());
  if (controlled.jacobian.cols() == reference.jacobian.cols() &&
    reference.jacobian.rows() >= 6 && controlled.jacobian.rows() >= 6)
  {
    const Eigen::Matrix3d ref_rotation_world_to_local = reference.pose.rotation().transpose();
    const Eigen::Vector3d relative_translation_world =
      controlled.pose.translation() - reference.pose.translation();
    const auto ref_linear = reference.jacobian.topRows(3);
    const auto ref_angular = reference.jacobian.bottomRows(3);
    const auto controlled_linear = controlled.jacobian.topRows(3);
    const auto controlled_angular = controlled.jacobian.bottomRows(3);

    full_jacobian.topRows(3) =
      ref_rotation_world_to_local *
      (controlled_linear - ref_linear + skew(relative_translation_world) * ref_angular);
    full_jacobian.bottomRows(3) =
      ref_rotation_world_to_local * (controlled_angular - ref_angular);
  }
  return full_jacobian;
}

Eigen::Matrix<double, 6, 1> EndEffectorPlanRelativePoseTask::relative_pose_error(
  const Eigen::Isometry3d & current_relative_pose) const
{
  Eigen::Matrix<double, 6, 1> full_error = Eigen::Matrix<double, 6, 1>::Zero();
  full_error.head<3>() =
    target_relative_pose_.translation() - current_relative_pose.translation();
  full_error.tail<3>() = quaternion_error(
    Eigen::Quaterniond(target_relative_pose_.rotation()).normalized(),
    Eigen::Quaterniond(current_relative_pose.rotation()).normalized());
  return full_error;
}

EndEffectorPlanRelativePoseTask::RelativeLinearization
EndEffectorPlanRelativePoseTask::evaluate_relative_linearization(
  const PlannerInput & input,
  const Eigen::VectorXd & q) const
{
  RelativeLinearization out;
  const auto active_indices = active_relative_indices();
  if (active_indices.empty()) {
    out.valid = true;
    out.relative_jacobian = Eigen::MatrixXd::Zero(0, q.size());
    out.active_error = Eigen::VectorXd::Zero(0);
    out.joint_jacobian = Eigen::MatrixXd::Zero(0, q.size());
    return out;
  }

  if (input.kinematics) {
    try {
      WholeBodyState state = input.state_template;
      state.joint_positions = q;
      state.joints_valid = true;
      const FrameState reference = input.kinematics->compute_frame_state(state, reference_frame_);
      const FrameState controlled = input.kinematics->compute_frame_state(state, controlled_frame_);
      out.current_relative_pose = reference.pose.inverse() * controlled.pose;
      out.relative_jacobian = relative_jacobian(reference, controlled);
      const auto full_error = relative_pose_error(out.current_relative_pose);
      out.active_error = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(active_indices.size()));
      Eigen::MatrixXd active_jacobian(
        static_cast<Eigen::Index>(active_indices.size()),
        out.relative_jacobian.cols());
      for (size_t i = 0; i < active_indices.size(); ++i) {
        active_jacobian.row(static_cast<Eigen::Index>(i)) = out.relative_jacobian.row(active_indices[i]);
        out.active_error(static_cast<Eigen::Index>(i)) = full_error(active_indices[i]);
      }
      if (active_jacobian.cols() < static_cast<Eigen::Index>(joint_names_.size())) {
        return out;
      }
      out.joint_jacobian = active_jacobian.rightCols(static_cast<Eigen::Index>(joint_names_.size()));
      out.valid = out.joint_jacobian.cols() == q.size();
      return out;
    } catch (const std::exception &) {
    }
  }

  if (input.relative_jacobian.cols() >= static_cast<Eigen::Index>(joint_names_.size()) &&
    input.q_current.size() == q.size())
  {
    out.current_relative_pose = input.current_relative_pose;
    out.relative_jacobian = input.relative_jacobian;
    Eigen::MatrixXd active_jacobian(
      static_cast<Eigen::Index>(active_indices.size()),
      input.relative_jacobian.cols());
    const auto full_error = relative_pose_error(input.current_relative_pose);
    out.active_error = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(active_indices.size()));
    for (size_t i = 0; i < active_indices.size(); ++i) {
      active_jacobian.row(static_cast<Eigen::Index>(i)) = input.relative_jacobian.row(active_indices[i]);
      out.active_error(static_cast<Eigen::Index>(i)) = full_error(active_indices[i]);
    }
    out.joint_jacobian = active_jacobian.rightCols(static_cast<Eigen::Index>(joint_names_.size()));
    out.active_error -= out.joint_jacobian * (q - input.q_current);
    out.valid = out.joint_jacobian.cols() == q.size();
  }
  return out;
}

std::vector<Eigen::Index> EndEffectorPlanRelativePoseTask::active_relative_indices() const
{
  std::vector<Eigen::Index> active_indices;
  active_indices.reserve(6);
  for (Eigen::Index i = 0; i < 6; ++i) {
    if (activation_(i)) {
      active_indices.push_back(i);
    }
  }
  return active_indices;
}

Eigen::VectorXd EndEffectorPlanRelativePoseTask::sample_trajectory(double elapsed) const
{
  if (!active_trajectory_.points.empty()) {
    if (active_trajectory_.points.size() == 1 || elapsed <= active_trajectory_.times.front()) {
      return active_trajectory_.points.front();
    }
    for (std::size_t i = 1; i < active_trajectory_.points.size(); ++i) {
      if (elapsed <= active_trajectory_.times[i]) {
        const double segment_duration = active_trajectory_.times[i] - active_trajectory_.times[i - 1];
        const double alpha = segment_duration <= 0.0 ?
          1.0 : std::clamp((elapsed - active_trajectory_.times[i - 1]) / segment_duration, 0.0, 1.0);
        return active_trajectory_.points[i - 1] +
          alpha * (active_trajectory_.points[i] - active_trajectory_.points[i - 1]);
      }
    }
    return active_trajectory_.points.back();
  }

  const double alpha = active_trajectory_.duration <= 0.0 ?
    1.0 : std::clamp(elapsed / active_trajectory_.duration, 0.0, 1.0);
  return active_trajectory_.start + alpha * (active_trajectory_.goal - active_trajectory_.start);
}

double EndEffectorPlanRelativePoseTask::joint_margin(const Eigen::VectorXd & q) const
{
  if (q.size() == 0 || lower_limits_.size() != q.size() || upper_limits_.size() != q.size()) {
    return 0.0;
  }
  double margin = std::numeric_limits<double>::infinity();
  for (Eigen::Index i = 0; i < q.size(); ++i) {
    const double range = upper_limits_(i) - lower_limits_(i);
    if (!std::isfinite(range) || range <= 0.0) {
      continue;
    }
    const double lower_margin = (q(i) - lower_limits_(i)) / range;
    const double upper_margin = (upper_limits_(i) - q(i)) / range;
    margin = std::min(margin, std::min(lower_margin, upper_margin));
  }
  return std::isfinite(margin) ? margin : 1.0;
}

double EndEffectorPlanRelativePoseTask::normalized_distance(
  const Eigen::VectorXd & a,
  const Eigen::VectorXd & b) const
{
  if (a.size() != b.size()) {
    return std::numeric_limits<double>::infinity();
  }
  Eigen::VectorXd normalized = Eigen::VectorXd::Zero(a.size());
  for (Eigen::Index i = 0; i < a.size(); ++i) {
    const double range = upper_limits_(i) - lower_limits_(i);
    normalized(i) = (std::isfinite(range) && range > 0.0) ?
      (a(i) - b(i)) / range : a(i) - b(i);
  }
  return normalized.norm();
}

double EndEffectorPlanRelativePoseTask::linearized_relative_error_norm(
  const Eigen::VectorXd & q,
  const Eigen::VectorXd & q_reference,
  const Eigen::MatrixXd & joint_jacobian,
  const Eigen::VectorXd & active_error) const
{
  if (q.size() != q_reference.size() || joint_jacobian.cols() != q.size()) {
    return std::numeric_limits<double>::infinity();
  }
  return (active_error - joint_jacobian * (q - q_reference)).norm();
}

bool EndEffectorPlanRelativePoseTask::within_limits(const Eigen::VectorXd & q) const
{
  if (lower_limits_.size() != q.size() || upper_limits_.size() != q.size()) {
    return false;
  }
  for (Eigen::Index i = 0; i < q.size(); ++i) {
    if (q(i) < lower_limits_(i) || q(i) > upper_limits_(i)) {
      return false;
    }
  }
  return true;
}

std::vector<Eigen::VectorXd> EndEffectorPlanRelativePoseTask::generate_sobol_seeds(std::size_t count) const
{
  std::vector<Eigen::VectorXd> seeds;
  const std::size_t dims = static_cast<std::size_t>(joint_names_.size());
  if (count == 0 || dims == 0) {
    return seeds;
  }
  seeds.reserve(count);
  const auto directions = sobol_direction_numbers(dims);
  std::vector<std::uint32_t> x(dims, 0U);
  constexpr double scale = 1.0 / 4294967296.0;
  const std::uint32_t scramble = static_cast<std::uint32_t>(global_seed_ * 0x9E3779B9U);

  for (std::size_t sample = 0; sample < count; ++sample) {
    const int bit = least_significant_zero_bit(static_cast<std::uint32_t>(sample));
    Eigen::VectorXd q = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(dims));
    for (std::size_t dim = 0; dim < dims; ++dim) {
      x[dim] ^= directions[dim][static_cast<std::size_t>(bit - 1)];
      const double unit = static_cast<double>(x[dim] ^ (scramble + static_cast<std::uint32_t>(dim * 7919U))) * scale;
      const double lower = lower_limits_(static_cast<Eigen::Index>(dim));
      const double upper = upper_limits_(static_cast<Eigen::Index>(dim));
      if (std::isfinite(lower) && std::isfinite(upper) && upper > lower) {
        q(static_cast<Eigen::Index>(dim)) = lower + std::clamp(unit, 0.0, 1.0) * (upper - lower);
      } else {
        q(static_cast<Eigen::Index>(dim)) = 0.0;
      }
    }
    seeds.push_back(q);
  }
  return seeds;
}

void EndEffectorPlanRelativePoseTask::dump_atlas_debug(
  const AtlasDebugData & debug_data,
  const PlannerInput & input) const
{
  if (!debug_) {
    return;
  }
  std::ofstream out(global_debug_dump_path_, std::ios::out | std::ios::trunc);
  if (!out.is_open()) {
    RCLCPP_WARN(
      rclcpp::get_logger("EndEffectorPlanRelativePoseTask"),
      "[%s] could not write manifold debug dump to %s",
      id_.c_str(),
      global_debug_dump_path_.c_str());
    return;
  }

  out << std::setprecision(9);
  out << "EndEffectorPlanRelativePoseTask manifold search debug dump\n";
  out << "task_id: " << id_ << "\n";
  out << "pipeline: Sobol -> Projection F(q)=0 -> valid solutions -> continuation/atlas"
      << " -> connectivity graph -> connected components\n";
  out << "debug_dump_path: " << global_debug_dump_path_ << "\n\n";

  out << "[target]\n";
  out << "target_translation: " << vector_to_string(input.target_relative_pose.translation()) << "\n";
  out << "current_relative_translation: " << vector_to_string(input.current_relative_pose.translation()) << "\n";
  out << "q_current: " << vector_to_string(input.q_current) << "\n";
  out << "active_error: " << vector_to_string(relative_pose_error(input.current_relative_pose)) << "\n\n";

  out << "[parameters]\n";
  out << "sobol_seed_count: " << debug_data.sobol_seeds.size() << "\n";
  out << "projection_iterations: " << global_projection_iterations_ << "\n";
  out << "relative_pose_tolerance: " << relative_pose_tolerance_
      << "  # shared default for projection, solution, reprojection and external goal tolerances\n";
  out << "projection_tolerance: " << global_projection_tolerance_
      << "  # DLS/Newton stops only when projected error is below this\n";
  out << "solution_tolerance: " << global_solution_tolerance_ << "\n";
  out << "reprojection_tolerance: " << reprojection_tolerance_ << "\n";
  out << "explore_seed_threshold: " << global_explore_threshold_
      << "  # loose near-manifold diagnostic threshold after max projection iterations\n";
  out << "deduplicate_radius: " << global_deduplicate_radius_ << "\n";
  out << "atlas_step: " << global_atlas_step_ << "\n";
  out << "atlas_neighbor_radius: " << global_atlas_neighbor_radius_ << "\n";
  out << "atlas_max_charts_per_seed: " << global_atlas_max_charts_per_seed_ << "\n";
  out << "atlas_max_total_charts: " << global_atlas_max_total_charts_ << "\n";
  out << "polar_angular_samples: " << global_polar_angular_samples_ << "\n";
  out << "polar_radial_step: " << global_polar_radial_step_ << "\n";
  out << "polar_max_radius: " << global_polar_max_radius_ << "\n\n";

  out << "[joint_limits]\n";
  for (size_t i = 0; i < joint_names_.size(); ++i) {
    const Eigen::Index index = static_cast<Eigen::Index>(i);
    out << joint_names_[i]
        << " lower=" << (index < lower_limits_.size() ? lower_limits_(index) : std::numeric_limits<double>::quiet_NaN())
        << " upper=" << (index < upper_limits_.size() ? upper_limits_(index) : std::numeric_limits<double>::quiet_NaN())
        << "\n";
  }
  out << "\n";

  out << "[summary]\n";
  const auto strict_projection_count = std::count_if(
    debug_data.projections.begin(),
    debug_data.projections.end(),
    [](const ProjectionResult & projection) {return projection.success;});
  const auto near_projection_count = std::count_if(
    debug_data.projections.begin(),
    debug_data.projections.end(),
    [](const ProjectionResult & projection) {return projection.near_explore_threshold;});
  out << "sobol_seeds: " << debug_data.sobol_seeds.size() << "\n";
  out << "projection_results: " << debug_data.projections.size() << "\n";
  out << "strict_projection_success: " << strict_projection_count << "\n";
  out << "near_explore_threshold: " << near_projection_count << "\n";
  out << "valid_candidates_after_atlas: " << debug_data.candidates.size() << "\n";
  out << "charts: " << debug_data.charts.size() << "\n";
  out << "graph_edges: " << debug_data.edges.size() << "\n";
  out << "components: " << debug_data.components.size() << "\n";
  out << "rejections: " << debug_data.rejection_log.size() << "\n\n";

  out << "[projection_iteration_trace]\n";
  out << "enabled: " << debug_data.projection_trace.enabled << "\n";
  if (debug_data.projection_trace.enabled) {
    out << "seed_index: S" << debug_data.projection_trace.seed_index << "\n";
    out << "seed: " << vector_to_string(debug_data.projection_trace.seed) << "\n";
    out << "clamped_seed: " << vector_to_string(debug_data.projection_trace.clamped_seed) << "\n";
    out << "seed_error_norm: " << debug_data.projection_trace.seed_error_norm << "\n";
    out << "seed_residual: " << vector_to_string(debug_data.projection_trace.seed_residual) << "\n";
    out << "explore_seed_threshold: " << debug_data.projection_trace.explore_seed_threshold << "\n";
    out << "projection_stop_rule: projected_error <= projection_tolerance\n";
    out << "candidate_decision_rule: projected_error <= min(projection_tolerance, solution_tolerance), "
        << "then sigma_min, joint_margin, collision_margin\n";
    out << "near_seed_diagnostic_rule: if projection stops by max iterations, "
        << "projected_error <= explore_seed_threshold means the seed got close but is not necessarily accepted\n";
    out << "iterations: " << debug_data.projection_trace.iterations.size() << "\n";
    for (const auto & iteration : debug_data.projection_trace.iterations) {
      out << "k=" << iteration.iteration
          << " error_norm=" << iteration.error_norm
          << " rank=" << iteration.rank
          << " q_k=" << vector_to_string(iteration.q)
          << " F_qk=" << vector_to_string(iteration.residual)
          << " e_p=" << vector_to_string(iteration.position_error)
          << " e_R=" << vector_to_string(iteration.orientation_error)
          << " delta_q=" << vector_to_string(iteration.delta_q)
          << " sigma_J=" << vector_to_string(iteration.singular_values)
          << " clamped_joints=[";
      for (std::size_t i = 0; i < iteration.clamped_joints.size(); ++i) {
        if (i > 0) {
          out << ", ";
        }
        out << joint_names_[i] << ":" << (iteration.clamped_joints[i] ? "1" : "0");
      }
      out << "]\n";
    }
  }
  out << "\n";

  out << "[sobol_seeds]\n";
  for (std::size_t i = 0; i < debug_data.sobol_seeds.size(); ++i) {
    out << "S" << i << " q=" << vector_to_string(debug_data.sobol_seeds[i]);
    if (i < debug_data.projections.size()) {
      const auto & projection = debug_data.projections[i];
      out << " projection_success=" << projection.success
          << " near_explore_threshold=" << projection.near_explore_threshold
          << " projection_error=" << projection.error_norm
          << " projection_iterations=" << projection.iterations
          << " projected_q=" << vector_to_string(projection.q);
    }
    out << "\n";
  }
  out << "\n";

  out << "[candidates]\n";
  for (std::size_t i = 0; i < debug_data.candidates.size(); ++i) {
    const auto & candidate = debug_data.candidates[i];
    out << "N" << i
        << " component=M" << candidate.component_id
        << " support=" << candidate.component_sample_count
        << " score=" << candidate.score
        << " component_score=" << candidate.component_score
        << " rel_error=" << candidate.relative_error_norm
        << " sigma_min=" << candidate.sigma_min
        << " manipulability=" << candidate.manipulability
        << " joint_margin=" << candidate.joint_margin
        << " collision_clearance=" << candidate.collision_margin
        << " nullspace_room=" << candidate.nullspace_room
        << " local_area=" << candidate.local_area
        << " boundary_radius=" << candidate.boundary_radius
        << " current_distance=" << candidate.current_distance
        << " curvature=" << candidate.curvature
        << " q=" << vector_to_string(candidate.q) << "\n";
  }
  out << "\n";

  out << "[charts]\n";
  for (const auto & chart : debug_data.charts) {
    out << "C" << chart.id
        << " node=N" << chart.candidate_index
        << " root=N" << chart.root_candidate_index
        << " component=M" << chart.component_id
        << " parent_chart=C" << chart.parent_chart_id
        << " depth=" << chart.depth
        << " expanded=" << chart.expanded
        << " tangent_dim=" << chart.tangent_basis.cols()
        << " chart_radius=" << chart.chart_radius
        << " local_area=" << chart.local_area
        << " boundary_radius=" << chart.boundary_radius
        << " curvature=" << chart.curvature << "\n";
    for (Eigen::Index col = 0; col < chart.tangent_basis.cols(); ++col) {
      out << "  tangent_" << col << "=" << vector_to_string(chart.tangent_basis.col(col)) << "\n";
    }
  }
  out << "\n";

  out << "[graph_edges]\n";
  for (std::size_t i = 0; i < debug_data.edges.size(); ++i) {
    const auto & edge = debug_data.edges[i];
    out << "E" << i
        << " N" << edge.from
        << " -- N" << edge.to
        << " distance=" << edge.distance
        << " reason=" << edge.reason << "\n";
  }
  out << "\n";

  out << "[components]\n";
  for (const auto & component : debug_data.components) {
    out << "M" << component.id
        << " samples=" << component.indices.size()
        << " representative=N" << component.representative_index
        << " score=" << component.score
        << " mean_sigma=" << component.mean_sigma
        << " min_sigma=" << component.min_sigma
        << " mean_manipulability=" << component.mean_manipulability
        << " mean_rel_error=" << component.mean_relative_error
        << " max_rel_error=" << component.max_relative_error
        << " mean_joint_margin=" << component.mean_joint_margin
        << " min_joint_margin=" << component.min_joint_margin
        << " mean_current_distance=" << component.mean_current_distance
        << " local_area_mean=" << component.local_area_mean
        << " boundary_radius_mean=" << component.boundary_radius_mean
        << " curvature_mean=" << component.curvature_mean << "\n";
    out << "  nodes=";
    for (std::size_t i = 0; i < component.indices.size(); ++i) {
      if (i > 0) {
        out << ", ";
      }
      out << "N" << component.indices[i];
    }
    out << "\n";
  }
  out << "\n";

  out << "[rejections]\n";
  for (const auto & rejection : debug_data.rejection_log) {
    out << rejection << "\n";
  }
}

std::string EndEffectorPlanRelativePoseTask::mode_name(Mode mode) const
{
  switch (mode) {
    case Mode::kIdle:
      return "Idle";
    case Mode::kNeedsPlan:
      return "NeedsPlan";
    case Mode::kPlanning:
      return "Planning";
    case Mode::kMoveToManifold:
      return "MoveToManifold";
    case Mode::kBlendToRelativePose:
      return "BlendToRelativePose";
    case Mode::kServoRelativePose:
      return "ServoRelativePose";
  }
  return "Unknown";
}

void EndEffectorPlanRelativePoseTask::debug_log(const std::string & message) const
{
  if (!debug_) {
    return;
  }
  std::ofstream out(global_debug_dump_path_, std::ios::out | std::ios::app);
  if (out.is_open()) {
    out << "[runtime_log][" << id_ << "] " << message << "\n";
  }
}

}  // namespace task_priority_kinematic_control

PLUGINLIB_EXPORT_CLASS(
  task_priority_kinematic_control::EndEffectorPlanRelativePoseTask,
  task_priority_kinematic_control::TaskBase)
