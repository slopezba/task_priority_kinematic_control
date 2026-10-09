#pragma once

#include "task_priority_kinematic_control/tasks/task_base.hpp"

#include <atomic>
#include <chrono>
#include <mutex>
#include <limits>
#include <string>
#include <thread>
#include <utility>

#include <rclcpp/executors/single_threaded_executor.hpp>
#include <rclcpp/node.hpp>
#include <rclcpp_action/rclcpp_action.hpp>
#include <sura_manipulation_actions/action/plan_bimanual_joint_trajectory.hpp>

namespace task_priority_kinematic_control
{

class EndEffectorPlanRelativePoseTask : public TaskBaseCommon
{
public:
  ~EndEffectorPlanRelativePoseTask() override;

  void configure(
    const std::string & id,
    const std::string & plugin_name,
    const std::map<std::string, rclcpp::Parameter> & parameters,
    const TaskContext & context) override;

  TaskComputation update(
    const WholeBodyState & state,
    const KinematicsBackend & backend) override;

  bool set_pose_goal(const geometry_msgs::msg::PoseStamped & goal) override;
  bool set_gain(const std::vector<double> & gain, std::string & message) override;
  void reset() override;
  msg::TaskStatus build_status() const override;
  std::vector<double> current_target() const override;

private:
  enum class Mode
  {
    kIdle,
    kNeedsPlan,
    kPlanning,
    kMoveToManifold,
    kBlendToRelativePose,
    kServoRelativePose
  };

  struct PlannerInput
  {
    WholeBodyState state_template;
    KinematicsBackendPtr kinematics;
    Eigen::VectorXd q_current;
    Eigen::Isometry3d target_relative_pose = Eigen::Isometry3d::Identity();
    Eigen::Isometry3d current_relative_pose = Eigen::Isometry3d::Identity();
    Eigen::MatrixXd relative_jacobian;
  };

  struct RelativeLinearization
  {
    bool valid = false;
    Eigen::Isometry3d current_relative_pose = Eigen::Isometry3d::Identity();
    Eigen::MatrixXd relative_jacobian;
    Eigen::VectorXd active_error;
    Eigen::MatrixXd joint_jacobian;
  };

  struct PlannedTrajectory
  {
    Eigen::VectorXd start;
    Eigen::VectorXd goal;
    std::vector<double> times;
    std::vector<Eigen::VectorXd> points;
    double duration = 0.0;
  };

  struct PlannerResult
  {
    bool success = false;
    bool timed_out = false;
    std::string message;
    Eigen::VectorXd q_selected;
    PlannedTrajectory trajectory;
    double score = 0.0;
  };

  struct Candidate
  {
    Eigen::VectorXd q;
    double relative_error_norm = 0.0;
    double sigma_min = 0.0;
    double manipulability = 0.0;
    double joint_margin = 0.0;
    double collision_margin = 0.0;
    double nullspace_room = 0.0;
    double local_area = 0.0;
    double boundary_radius = 0.0;
    double smoothness = 0.0;
    double current_distance = 0.0;
    double curvature = 0.0;
    double score = 0.0;
    int component_id = -1;
    int component_sample_count = 1;
    double component_score = 0.0;
  };

  struct ProjectionResult
  {
    bool success = false;
    bool near_explore_threshold = false;
    Eigen::VectorXd q;
    double error_norm = std::numeric_limits<double>::infinity();
    int iterations = 0;
  };

  struct ProjectionIterationTrace
  {
    int iteration = 0;
    Eigen::VectorXd q;
    Eigen::VectorXd residual;
    Eigen::Vector3d position_error = Eigen::Vector3d::Zero();
    Eigen::Vector3d orientation_error = Eigen::Vector3d::Zero();
    Eigen::VectorXd delta_q;
    Eigen::VectorXd singular_values;
    int rank = 0;
    std::vector<bool> clamped_joints;
    double error_norm = std::numeric_limits<double>::infinity();
  };

  struct ProjectionTrace
  {
    bool enabled = false;
    std::size_t seed_index = 0;
    Eigen::VectorXd seed;
    Eigen::VectorXd clamped_seed;
    Eigen::VectorXd seed_residual;
    double seed_error_norm = std::numeric_limits<double>::infinity();
    double explore_seed_threshold = 0.0;
    std::vector<ProjectionIterationTrace> iterations;
  };

  struct Component
  {
    int id = -1;
    std::vector<std::size_t> indices;
    double score = 0.0;
    double mean_sigma = 0.0;
    double min_sigma = 0.0;
    double mean_manipulability = 0.0;
    double mean_relative_error = 0.0;
    double max_relative_error = 0.0;
    double mean_joint_margin = 0.0;
    double min_joint_margin = 0.0;
    double mean_current_distance = 0.0;
    double local_area_mean = 0.0;
    double boundary_radius_mean = 0.0;
    double curvature_mean = 0.0;
    std::size_t representative_index = 0;
  };

  struct AtlasChart
  {
    std::size_t id = 0;
    std::size_t candidate_index = 0;
    std::size_t root_candidate_index = 0;
    int component_id = -1;
    std::size_t parent_chart_id = 0;
    int depth = 0;
    Eigen::MatrixXd tangent_basis;
    double chart_radius = 0.0;
    double local_area = 0.0;
    double boundary_radius = 0.0;
    double curvature = 0.0;
    bool expanded = false;
  };

  struct AtlasEdge
  {
    std::size_t from = 0;
    std::size_t to = 0;
    double distance = 0.0;
    std::string reason;
  };

  struct AtlasDebugData
  {
    std::vector<Eigen::VectorXd> sobol_seeds;
    std::vector<ProjectionResult> projections;
    ProjectionTrace projection_trace;
    std::vector<Candidate> candidates;
    std::vector<AtlasChart> charts;
    std::vector<AtlasEdge> edges;
    std::vector<Component> components;
    std::vector<std::string> rejection_log;
  };

  void stop_planner();
  void start_planning_locked(const PlannerInput & input);
  void planner_main(PlannerInput input);
  PlannerResult plan(const PlannerInput & input, const std::chrono::steady_clock::time_point & deadline);
  std::vector<Candidate> explore_candidates(
    const PlannerInput & input,
    const std::chrono::steady_clock::time_point & deadline) const;
  std::vector<Candidate> explore_local_candidates(
    const PlannerInput & input,
    const std::chrono::steady_clock::time_point & deadline) const;
  std::vector<Candidate> explore_global_candidates(
    const PlannerInput & input,
    const std::chrono::steady_clock::time_point & deadline) const;
  ProjectionResult project_to_manifold(
    const Eigen::VectorXd & seed,
    const PlannerInput & input,
    const Eigen::MatrixXd & joint_jacobian,
    const Eigen::VectorXd & active_error,
    const std::chrono::steady_clock::time_point & deadline,
    ProjectionTrace * trace = nullptr) const;
  std::vector<Component> cluster_candidates(
    std::vector<Candidate> & candidates,
    const PlannerInput & input,
    const Eigen::MatrixXd & joint_jacobian,
    const Eigen::VectorXd & active_error,
    const Eigen::MatrixXd & nullspace,
    const std::chrono::steady_clock::time_point & deadline) const;
  std::vector<Component> build_atlas_components(
    std::vector<Candidate> & candidates,
    const PlannerInput & input,
    const Eigen::MatrixXd & joint_jacobian,
    const Eigen::VectorXd & active_error,
    AtlasDebugData & debug_data,
    const std::chrono::steady_clock::time_point & deadline) const;
  bool validate_edge(
    const Candidate & a,
    const Candidate & b,
    const PlannerInput & input,
    const Eigen::MatrixXd & joint_jacobian,
    const Eigen::VectorXd & active_error,
    const Eigen::MatrixXd & nullspace,
    const std::chrono::steady_clock::time_point & deadline) const;
  void estimate_local_chart(Candidate & candidate, const Eigen::MatrixXd & nullspace) const;
  void estimate_local_chart(
    Candidate & candidate,
    const PlannerInput & input,
    const Eigen::MatrixXd & joint_jacobian,
    const Eigen::VectorXd & active_error,
    const Eigen::MatrixXd & nullspace,
    const std::chrono::steady_clock::time_point & deadline) const;
  bool validate_candidate(const Candidate & candidate) const;
  bool validate_trajectory(const Eigen::VectorXd & start, const Eigen::VectorXd & goal) const;
  bool request_external_trajectory(
    const Eigen::VectorXd & start,
    const Eigen::VectorXd & goal,
    const std::chrono::steady_clock::time_point & deadline,
    PlannedTrajectory & trajectory,
    std::string & message,
    bool & timed_out);
  bool trajectory_from_msg(
    const trajectory_msgs::msg::JointTrajectory & msg,
    const Eigen::VectorXd & start,
    const Eigen::VectorXd & goal,
    PlannedTrajectory & trajectory,
    std::string & message) const;
  Candidate score_candidate(
    const Eigen::VectorXd & q,
    const PlannerInput & input,
    const Eigen::MatrixXd & nullspace,
    double relative_error_norm = 0.0) const;
  Candidate score_candidate(
    const Eigen::VectorXd & q,
    const PlannerInput & input,
    const Eigen::MatrixXd & nullspace,
    const Eigen::MatrixXd & relative_jacobian,
    double relative_error_norm = 0.0) const;

  TaskComputation move_to_manifold_update(const WholeBodyState & state, bool allow_transition);
  TaskComputation blend_update(
    const WholeBodyState & state,
    const KinematicsBackend & backend);
  TaskComputation relative_pose_update(const KinematicsBackend & backend, double row_scale) const;

  Eigen::MatrixXd relative_jacobian(
    const FrameState & reference,
    const FrameState & controlled) const;
  RelativeLinearization evaluate_relative_linearization(
    const PlannerInput & input,
    const Eigen::VectorXd & q) const;
  Eigen::Matrix<double, 6, 1> relative_pose_error(const Eigen::Isometry3d & current_relative_pose) const;
  std::vector<Eigen::Index> active_relative_indices() const;
  Eigen::VectorXd sample_trajectory(double elapsed) const;
  double joint_margin(const Eigen::VectorXd & q) const;
  double normalized_distance(const Eigen::VectorXd & a, const Eigen::VectorXd & b) const;
  double linearized_relative_error_norm(
    const Eigen::VectorXd & q,
    const Eigen::VectorXd & q_reference,
    const Eigen::MatrixXd & joint_jacobian,
    const Eigen::VectorXd & active_error) const;
  bool within_limits(const Eigen::VectorXd & q) const;
  std::vector<Eigen::VectorXd> generate_sobol_seeds(std::size_t count) const;
  void dump_atlas_debug(const AtlasDebugData & debug_data, const PlannerInput & input) const;
  std::string mode_name(Mode mode) const;
  void debug_log(const std::string & message) const;

  std::string reference_frame_;
  std::string controlled_frame_;
  Eigen::Matrix<double, 6, 1> gains_ = Eigen::Matrix<double, 6, 1>::Ones();
  Eigen::Matrix<bool, 6, 1> activation_ = Eigen::Matrix<bool, 6, 1>::Constant(true);
  Eigen::Isometry3d target_relative_pose_ = Eigen::Isometry3d::Identity();
  bool has_target_relative_pose_ = false;

  std::vector<std::string> joint_names_;
  Eigen::VectorXd lower_limits_;
  Eigen::VectorXd upper_limits_;

  double planning_timeout_ = 5.0;
  double move_gain_ = 0.8;
  double move_goal_tolerance_ = 0.04;
  double blend_duration_ = 2.0;
  double max_joint_velocity_ = 0.25;
  double min_sigma_ = 0.03;
  double min_joint_margin_ = 0.02;
  double min_collision_margin_ = 0.04;
  double nullspace_probe_radius_ = 0.12;
  int nullspace_probe_steps_ = 5;
  int reprojection_iterations_ = 20;
  double relative_pose_tolerance_ = 1.0e-4;
  double reprojection_tolerance_ = 1.0e-4;

  double weight_sigma_ = 1.0;
  double weight_collision_ = 2.0;
  double weight_limits_ = 1.0;
  double weight_room_ = 1.5;
  double weight_smoothness_ = 0.5;
  double weight_current_distance_ = 1.2;
  double weight_curvature_ = 0.5;
  double weight_boundary_ = 0.15;
  double weight_support_ = 0.15;
  double weight_relative_error_ = 5.0;
  bool debug_ = false;
  bool fallback_to_servo_on_plan_failure_ = true;
  bool global_search_enabled_ = false;
  int global_random_starts_ = 128;
  int global_local_perturbation_starts_ = 12;
  unsigned int global_seed_ = 7;
  int global_projection_iterations_ = 20;
  double global_projection_tolerance_ = 1.0e-4;
  double global_solution_tolerance_ = 5.0e-4;
  double global_explore_threshold_ = 0.04;
  double global_deduplicate_radius_ = 0.004;
  double global_connect_radius_ = 1.25;
  double global_max_edge_projection_distance_ = 0.08;
  int global_max_edge_neighbors_ = 2;
  int global_path_check_steps_ = 2;
  int global_max_components_to_plan_ = 2;
  int global_polar_angular_samples_ = 8;
  double global_polar_radial_step_ = 0.06;
  double global_polar_max_radius_ = 0.36;
  int global_max_charts_per_component_ = 3;
  double global_max_projector_change_ = 0.35;
  int global_atlas_max_charts_per_seed_ = 3;
  int global_atlas_max_total_charts_ = 160;
  double global_atlas_step_ = 0.08;
  double global_atlas_neighbor_radius_ = 0.12;
  bool global_projection_trace_enabled_ = true;
  int global_projection_trace_seed_index_ = 0;
  std::string global_debug_dump_path_ = "/tmp/end_effector_plan_relative_pose_manifold_debug.txt";
  double preferred_joint_margin_ = 0.08;
  bool external_planner_enabled_ = false;
  std::string external_planner_action_name_ = "/cirtesub/manipulation/plan_bimanual_joint_trajectory";
  double external_planner_wait_timeout_ = 1.0;
  double external_planner_result_timeout_ = 5.0;
  double external_planner_goal_tolerance_ = 0.02;

  using PlanBimanualJointTrajectory =
    sura_manipulation_actions::action::PlanBimanualJointTrajectory;
  using GoalHandlePlanBimanualJointTrajectory =
    rclcpp_action::ClientGoalHandle<PlanBimanualJointTrajectory>;
  rclcpp::Node::SharedPtr external_planner_node_;
  std::shared_ptr<rclcpp::executors::SingleThreadedExecutor> external_planner_executor_;
  std::thread external_planner_spin_thread_;
  rclcpp_action::Client<PlanBimanualJointTrajectory>::SharedPtr external_planner_client_;

  mutable std::mutex mutex_;
  Mode mode_ = Mode::kIdle;
  std::string status_message_ = "idle";
  PlannedTrajectory active_trajectory_;
  std::chrono::steady_clock::time_point trajectory_start_;
  std::chrono::steady_clock::time_point blend_start_;
  Eigen::VectorXd current_joint_target_;
  double last_plan_score_ = 0.0;

  std::thread planner_thread_;
  std::atomic_bool planner_cancel_{false};
  bool planner_running_ = false;
  bool planner_result_ready_ = false;
  PlannerResult planner_result_;
};

}  // namespace task_priority_kinematic_control
