#include "task_priority_kinematic_control/core/capsule_observer.hpp"
#include <Eigen/Geometry>
#include <algorithm>
#include <cmath>
#include <limits>
#include <rcl_interfaces/msg/parameter_descriptor.hpp>
#include <rcl_interfaces/msg/set_parameters_result.hpp>

namespace task_priority_kinematic_control
{
namespace
{
using Marker = visualization_msgs::msg::Marker;
Marker marker(const ManualCapsule & definition, const std::string & frame, int id, int type)
{
  Marker m;
  m.header.frame_id = frame; m.ns = "self_collision_avoidance/" + definition.name;
  m.id = id; m.type = type; m.action = Marker::ADD;
  m.pose.orientation.w = 1.0;
  m.scale.x = m.scale.y = m.scale.z = 2 * definition.radius;
  m.color.a = 0.55f;
  if (definition.start_frame.find("alpha_right") != std::string::npos) {
    m.color.b = 1.0f; m.color.g = 0.2f;
  } else if (definition.start_frame.find("alpha_left") != std::string::npos) {
    m.color.g = 1.0f; m.color.b = 0.2f;
  } else {m.color.r = m.color.g = m.color.b = 0.6f;}
  return m;
}
void position(Marker & marker, const Eigen::Vector3d & p)
{marker.pose.position.x = p.x(); marker.pose.position.y = p.y(); marker.pose.position.z = p.z();}
}

visualization_msgs::msg::MarkerArray capsule_markers(
  const std::vector<ManualCapsule> & definitions, const CollisionSnapshot & snapshot,
  const std::string & frame, double length_epsilon,
  const std::vector<ManualPlane> & planes)
{
  visualization_msgs::msg::MarkerArray array;
  array.markers.reserve(3 * definitions.size() + planes.size());
  for (size_t i = 0; i < definitions.size(); ++i) {
    auto cylinder = marker(definitions[i], frame, 0, Marker::CYLINDER);
    auto start = marker(definitions[i], frame, 1, Marker::SPHERE);
    auto end = marker(definitions[i], frame, 2, Marker::SPHERE);
    if (i >= snapshot.capsules.size() || !snapshot.capsules[i].valid) {
      cylinder.action = start.action = end.action = Marker::DELETE;
    } else {
      const auto & capsule = snapshot.capsules[i];
      position(start, capsule.a); position(end, capsule.b);
      position(cylinder, 0.5 * capsule.a + 0.5 * capsule.b);
      const Eigen::Vector3d axis = capsule.b - capsule.a;
      const double length = axis.stableNorm();
      if (length <= length_epsilon) {
        cylinder.action = end.action = Marker::DELETE;
      } else {
        cylinder.scale.z = length;
        const Eigen::Quaterniond orientation = Eigen::Quaterniond::FromTwoVectors(Eigen::Vector3d::UnitZ(), axis / length);
        cylinder.pose.orientation.x = orientation.x(); cylinder.pose.orientation.y = orientation.y();
        cylinder.pose.orientation.z = orientation.z(); cylinder.pose.orientation.w = orientation.w();
      }
      if (capsule.highlighted) {
        for (auto * m : {&cylinder, &start, &end}) {m->color.r = 1; m->color.g = m->color.b = 0;}
      }
    }
    for (auto * m : {&cylinder, &start, &end}) {m->header.stamp = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(snapshot.timestamp_ns));}
    array.markers.push_back(std::move(cylinder)); array.markers.push_back(std::move(start)); array.markers.push_back(std::move(end));
  }
  for (size_t i = 0; i < planes.size(); ++i) {
    Marker m;
    m.header.frame_id = frame;
    m.header.stamp = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(snapshot.timestamp_ns));
    m.ns = "self_collision_avoidance/planes/" + planes[i].name;
    m.id = 0; m.type = Marker::TRIANGLE_LIST; m.action = Marker::ADD;
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = m.scale.z = 1.0;
    m.color.r = 1.0f; m.color.g = 0.7f; m.color.b = 0.1f; m.color.a = 0.35f;
    if (i >= snapshot.planes.size() || !snapshot.planes[i].valid) {
      m.action = Marker::DELETE;
    } else {
      const auto & plane = snapshot.planes[i];
      // Both windings keep the zero-thickness rectangle visible from either side.
      for (const auto corner : {0, 1, 2, 0, 2, 3, 2, 1, 0, 3, 2, 0}) {
        geometry_msgs::msg::Point point;
        point.x = plane.corners[corner].x(); point.y = plane.corners[corner].y();
        point.z = plane.corners[corner].z(); m.points.push_back(point);
      }
      if (plane.highlighted) {m.color.g = m.color.b = 0.0f;}
    }
    array.markers.push_back(std::move(m));
  }
  return array;
}

visualization_msgs::msg::MarkerArray delete_capsule_markers(
  const std::vector<ManualCapsule> & definitions, const std::string & frame,
  const std::vector<ManualPlane> & planes)
{
  CollisionSnapshot snapshot;
  return capsule_markers(definitions, snapshot, frame, 1e-9, planes);
}

visualization_msgs::msg::MarkerArray joint_limits_markers(
  const std::vector<std::string> & joint_names, const JointLimitsSnapshot & snapshot,
  const std::string & frame, const std::string & task_id)
{
  visualization_msgs::msg::MarkerArray array;
  array.markers.reserve(joint_names.size());
  for (size_t i = 0; i < joint_names.size(); ++i) {
    Marker m;
    m.header.frame_id = frame;
    m.header.stamp = static_cast<builtin_interfaces::msg::Time>(rclcpp::Time(snapshot.timestamp_ns));
    m.ns = task_id + "/limits/" + joint_names[i];
    m.id = static_cast<int>(i); m.type = Marker::CYLINDER; m.action = Marker::DELETE;
    m.pose.orientation.w = 1.0;
    m.scale.x = m.scale.y = 0.06; m.scale.z = 0.07;
    m.color.r = m.color.b = m.color.a = 1.0f;
    if (snapshot.enabled && snapshot.visualization_enabled && i < snapshot.joints.size() &&
      snapshot.joints[i].active && snapshot.joints[i].valid) {
      m.action = Marker::ADD;
      position(m, snapshot.joints[i].position);
      const Eigen::Quaterniond orientation = Eigen::Quaterniond::FromTwoVectors(
        Eigen::Vector3d::UnitZ(), snapshot.joints[i].axis);
      m.pose.orientation.x = orientation.x(); m.pose.orientation.y = orientation.y();
      m.pose.orientation.z = orientation.z(); m.pose.orientation.w = orientation.w();
    }
    array.markers.push_back(std::move(m));
  }
  return array;
}

visualization_msgs::msg::MarkerArray delete_joint_limits_markers(
  const std::vector<std::string> & joint_names, const std::string & frame, const std::string & task_id)
{
  return joint_limits_markers(joint_names, JointLimitsSnapshot{}, frame, task_id);
}

CapsuleObserver::CapsuleObserver(const std::vector<std::shared_ptr<TaskBase>> & tasks,
  const std::string & host_name, const rclcpp::Context::SharedPtr & context)
{
  const auto slash = host_name.find_last_of('/');
  rclcpp::NodeOptions options;
  options.context(context).use_global_arguments(false);
  node_ = std::make_shared<rclcpp::Node>(host_name.substr(slash + 1) + "_collision_observer",
    slash == 0 ? "/" : host_name.substr(0, slash), options);
  double rate = 1.0;  // Diagnostics run even with visualization disabled.
  for (const auto & task : tasks) {
    auto collision = std::dynamic_pointer_cast<SelfCollisionAvoidanceTask>(task);
    if (collision) {
      Entry entry;
      entry.task = collision;
      entry.publisher = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
        host_name + "/" + task->id() + "/capsules", rclcpp::QoS(1).reliable().durability_volatile());
      rate = std::max(rate, collision->capsule_publish_rate());
      entries_.push_back(std::move(entry));
    }
    auto limits = std::dynamic_pointer_cast<JointLimitsTask>(task);
    if (limits) {
      LimitsEntry entry;
      entry.task = limits;
      entry.publisher = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
        host_name + "/" + task->id() + "/limits_markers", rclcpp::QoS(1).reliable().durability_volatile());
      rate = std::max(rate, limits->limits_marker_publish_rate());
      limits_entries_.push_back(std::move(entry));
    }
    if (auto tilt = std::dynamic_pointer_cast<StabilizeLimitsTask>(task)) {
      TiltLimitsEntry entry;
      entry.task = tilt;
      entry.renderer = std::make_unique<StabilizeLimitsMarkerRenderer>(
        tilt->base_frame(), tilt->id(), tilt->publish_marker_rate());
      entry.publish_period = std::chrono::nanoseconds(
        static_cast<int64_t>(1e9 / tilt->publish_marker_rate()));
      entry.next_publish = std::chrono::steady_clock::now();
      entry.publisher = node_->create_publisher<visualization_msgs::msg::MarkerArray>(
        host_name + "/" + task->id() + "/limits_markers", rclcpp::QoS(1).reliable().durability_volatile());
      rate = std::max(rate, tilt->publish_marker_rate());
      tilt_limits_entries_.push_back(std::move(entry));
    }
  }
  if (entries_.empty() && limits_entries_.empty() && tilt_limits_entries_.empty()) {return;}
  rclcpp::ExecutorOptions executor_options;
  executor_options.context = context;
  executor_ = std::make_shared<rclcpp::executors::SingleThreadedExecutor>(executor_options);
  executor_->add_node(node_);
  timer_ = node_->create_wall_timer(std::chrono::duration_cast<std::chrono::nanoseconds>(
    std::chrono::duration<double>(1.0 / rate)), [this] {tick();});
  worker_ = std::thread([this] {
      try {executor_->spin();}
      catch (const std::exception & ex) {RCLCPP_ERROR(node_->get_logger(), "Capsule observer stopped: %s", ex.what());}
    });
}

CapsuleObserver::~CapsuleObserver()
{
  if (timer_) {timer_->cancel();}
  if (executor_) {executor_->cancel();}
  if (worker_.joinable()) {worker_.join();}
  // Lifecycle cleanup may wait on DDS; the control update never calls this destructor.
  for (auto & entry : entries_) {
    if (entry.visible && rclcpp::ok(node_->get_node_base_interface()->get_context())) {
      try {entry.publisher->publish(delete_capsule_markers(entry.task->capsules(), entry.task->base_frame(), entry.task->planes()));}
      catch (const std::exception &) {}
    }
  }
  for (auto & entry : limits_entries_) {
    if (entry.visible && rclcpp::ok(node_->get_node_base_interface()->get_context())) {
      try {
        entry.publisher->publish(delete_joint_limits_markers(
          entry.task->joint_names(), entry.task->base_frame(), entry.task->id()));
      } catch (const std::exception &) {}
    }
  }
  for (auto & entry : tilt_limits_entries_) {
    if (entry.visible && rclcpp::ok(node_->get_node_base_interface()->get_context())) {
      try {entry.publisher->publish(entry.renderer->clear());}
      catch (const std::exception &) {}
    }
  }
}

void CapsuleObserver::tick()
{
  const auto now = std::chrono::steady_clock::now();
  for (auto & entry : entries_) {
    if (const auto * next = entry.task->consume_snapshot()) {entry.snapshot = next;}
    if (entry.snapshot) {
      const auto & metrics = entry.snapshot->metrics;
      if (metrics.stop_arms && (!entry.previously_stopped || now - entry.last_warning >= std::chrono::seconds(1))) {
        const auto pair = entry.task->collision_pair_name(metrics.fault_pair);
        RCLCPP_WARN(node_->get_logger(), "Collision geometry error: arm velocities set to zero; pair=%s degenerate_pairs=%zu invalid_geometry=%d",
          pair.c_str(), metrics.degenerate_pairs, metrics.invalid_geometry);
        entry.last_warning = now;
      } else if (!metrics.stop_arms && entry.previously_stopped) {
        RCLCPP_INFO(node_->get_logger(), "Collision geometry recovered; arm motion can resume");
      }
      entry.previously_stopped = metrics.stop_arms;
    }
    const bool display = entry.task->publish_capsules() && entry.snapshot && entry.snapshot->metrics.enabled;
    if (!display) {
      if (entry.visible) {
        entry.publisher->publish(delete_capsule_markers(entry.task->capsules(), entry.task->base_frame(), entry.task->planes()));
        entry.visible = false;
      }
      continue;
    }
    const double rate = entry.task->capsule_publish_rate();
    if (rate <= 0 || now - entry.last_publish < std::chrono::duration<double>(1.0 / rate)) {continue;}
    entry.publisher->publish(capsule_markers(entry.task->capsules(), *entry.snapshot,
      entry.task->base_frame(), entry.task->segment_length_epsilon(), entry.task->planes()));
    entry.last_publish = now; entry.visible = true;
  }
  for (auto & entry : limits_entries_) {
    if (const auto * next = entry.task->consume_snapshot()) {entry.snapshot = next;}
    const bool display = entry.task->publish_limits_marker() && entry.snapshot &&
      entry.snapshot->visualization_state == entry.task->marker_visibility_state() &&
      entry.snapshot->enabled && entry.snapshot->visualization_enabled &&
      std::any_of(entry.snapshot->joints.begin(), entry.snapshot->joints.end(),
        [](const JointLimitPose & joint) {return joint.active && joint.valid;});
    if (!display) {
      if (entry.visible) {
        entry.publisher->publish(delete_joint_limits_markers(
          entry.task->joint_names(), entry.task->base_frame(), entry.task->id()));
        entry.visible = false;
      }
      continue;
    }
    if (now - entry.last_publish < std::chrono::duration<double>(1.0 / entry.task->limits_marker_publish_rate())) {
      continue;
    }
    entry.publisher->publish(joint_limits_markers(
      entry.task->joint_names(), *entry.snapshot, entry.task->base_frame(), entry.task->id()));
    entry.last_publish = now; entry.visible = true;
  }
  for (auto & entry : tilt_limits_entries_) {
    if (const auto * next = entry.task->consume_snapshot()) {entry.snapshot = next;}
    const bool display = entry.task->debug() && entry.snapshot &&
      entry.snapshot->visualization_state == entry.task->marker_visibility_state() &&
      entry.snapshot->visualization_enabled && entry.snapshot->valid &&
      now - entry.snapshot->captured <= std::chrono::duration<double>(
        std::max(1.0, 2.0 / entry.task->publish_marker_rate()));
    if (!display) {
      if (entry.visible) {
        entry.publisher->publish(entry.renderer->clear());
        entry.visible = false;
      }
      continue;
    }
    if (now < entry.next_publish) {continue;}
    entry.publisher->publish(entry.renderer->update(*entry.snapshot));
    // Advance a fixed schedule, skipping missed periods. Timer jitter must not
    // halve a 2 Hz stream by repeatedly missing a relative 500 ms threshold.
    entry.next_publish += entry.publish_period * (1 + (now - entry.next_publish) / entry.publish_period);
    entry.visible = true;
  }
}

void declare_joint_limits_marker_parameters(
  const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr & parameters,
  const std::string & prefix)
{
  if (!parameters->has_parameter(prefix + "publish_limits_marker")) {
    parameters->declare_parameter(prefix + "publish_limits_marker", rclcpp::ParameterValue(false),
      rcl_interfaces::msg::ParameterDescriptor{});
  }
  if (!parameters->has_parameter(prefix + "limits_marker_publish_rate")) {
    parameters->declare_parameter(prefix + "limits_marker_publish_rate", rclcpp::ParameterValue(1.0),
      rcl_interfaces::msg::ParameterDescriptor{});
  }
}

void declare_stabilize_limits_debug_parameters(
  const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr & parameters,
  const std::string & prefix)
{
  if (!parameters->has_parameter(prefix + "debug")) {
    parameters->declare_parameter(prefix + "debug", rclcpp::ParameterValue(false),
      rcl_interfaces::msg::ParameterDescriptor{});
  }
  if (!parameters->has_parameter(prefix + "publish_marker_rate")) {
    parameters->declare_parameter(prefix + "publish_marker_rate", rclcpp::ParameterValue(2.0),
      rcl_interfaces::msg::ParameterDescriptor{});
  }
}

rcl_interfaces::msg::SetParametersResult validate_joint_limits_marker_parameter_updates(
  const std::vector<rclcpp::Parameter> & parameters,
  const std::vector<std::shared_ptr<TaskBase>> & tasks, bool apply_visualization)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto & task : tasks) {
    const auto limits = std::dynamic_pointer_cast<JointLimitsTask>(task);
    const auto tilt = std::dynamic_pointer_cast<StabilizeLimitsTask>(task);
    if (!limits && !tilt) {continue;}
    const std::string prefix = "tasks." + task->id() + ".";
    const std::string toggle = tilt ? "debug" : "publish_limits_marker";
    const std::string rate = tilt ? "publish_marker_rate" : "limits_marker_publish_rate";
    for (const auto & parameter : parameters) {
      if (parameter.get_name() == prefix + toggle &&
        parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL) {
        result.successful = false; result.reason = toggle + " requires bool"; return result;
      }
      if (parameter.get_name() == prefix + rate) {
        result.successful = false;
        result.reason = parameter.get_name() + " requires reloading the controller";
        return result;
      }
    }
  }
  // Apply only after every visualization parameter in this transaction has been validated.
  if (apply_visualization) {
    for (const auto & task : tasks) {
      const auto limits = std::dynamic_pointer_cast<JointLimitsTask>(task);
      const auto tilt = std::dynamic_pointer_cast<StabilizeLimitsTask>(task);
      if (!limits && !tilt) {continue;}
      const std::string toggle = tilt ? "debug" : "publish_limits_marker";
      for (const auto & parameter : parameters) {
        if (parameter.get_name() == "tasks." + task->id() + "." + toggle) {
          if (limits) {limits->set_publish_limits_marker(parameter.as_bool());}
          else {tilt->set_debug(parameter.as_bool());}
        }
      }
    }
  }
  return result;
}

void declare_collision_parameters(
  const rclcpp::node_interfaces::NodeParametersInterface::SharedPtr & parameters,
  const std::string & prefix)
{
  auto declare = [&](const std::string & name, const rclcpp::ParameterValue & value) {
      if (!parameters->has_parameter(prefix + name)) {
        parameters->declare_parameter(prefix + name, value, rcl_interfaces::msg::ParameterDescriptor{});
      }
    };
  declare("safe_distance", rclcpp::ParameterValue(0.05));
  declare("activation_distance", rclcpp::ParameterValue(0.12));
  rclcpp::Parameter activation;
  parameters->get_parameter(prefix + "activation_distance", activation);
  declare("release_distance", rclcpp::ParameterValue(activation.as_double() + 0.01));
  declare("eps", rclcpp::ParameterValue(0.0001));
  declare("max_repulsive_velocity", rclcpp::ParameterValue(0.08));
  declare("parallel_epsilon", rclcpp::ParameterValue(1e-8));
  declare("segment_length_epsilon", rclcpp::ParameterValue(1e-9));
  declare("degenerate_axis_distance", rclcpp::ParameterValue(1e-6));
  declare("publish_capsules", rclcpp::ParameterValue(true));
  declare("capsule_publish_rate", rclcpp::ParameterValue(1.0));
  declare("capsule_names", rclcpp::ParameterValue(std::vector<std::string>{}));
  declare("plane_names", rclcpp::ParameterValue(std::vector<std::string>{}));
  declare("ignored_collision_pairs", rclcpp::ParameterValue(std::vector<std::string>{}));
  rclcpp::Parameter names;
  parameters->get_parameter(prefix + "capsule_names", names);
  for (const auto & name : names.as_string_array()) {
    declare("capsules." + name + ".start_frame", rclcpp::ParameterValue(std::string{}));
    declare("capsules." + name + ".end_frame", rclcpp::ParameterValue(std::string{}));
    declare("capsules." + name + ".radius", rclcpp::ParameterValue(0.0));
  }
  parameters->get_parameter(prefix + "plane_names", names);
  for (const auto & name : names.as_string_array()) {
    const auto plane_prefix = "planes." + name + ".";
    declare(plane_prefix + "reference_frame", rclcpp::ParameterValue(std::string{}));
    declare(plane_prefix + "normal_axis", rclcpp::ParameterValue(std::string{}));
    declare(plane_prefix + "position", rclcpp::ParameterValue(std::numeric_limits<double>::quiet_NaN()));
    declare(plane_prefix + "allowed_side", rclcpp::ParameterValue(std::string{}));
    declare(plane_prefix + "bounds_min", rclcpp::ParameterValue(std::vector<double>{}));
    declare(plane_prefix + "bounds_max", rclcpp::ParameterValue(std::vector<double>{}));
    declare(plane_prefix + "checked_capsules", rclcpp::ParameterValue(std::vector<std::string>{}));
  }
}

rcl_interfaces::msg::SetParametersResult validate_collision_parameter_updates(
  const std::vector<rclcpp::Parameter> & parameters,
  const std::vector<std::shared_ptr<TaskBase>> & tasks, bool apply_visualization)
{
  rcl_interfaces::msg::SetParametersResult result;
  result.successful = true;
  for (const auto & task : tasks) {
    const auto collision = std::dynamic_pointer_cast<SelfCollisionAvoidanceTask>(task);
    if (!collision) {continue;}
    const std::string prefix = "tasks." + task->id() + ".";
    for (const auto & parameter : parameters) {
      if (parameter.get_name().rfind(prefix, 0) != 0) {continue;}
      const auto field = parameter.get_name().substr(prefix.size());
      if (field == "publish_capsules") {
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_BOOL ||
          (parameter.as_bool() && collision->capsule_publish_rate() <= 0)) {
          result.successful = false; result.reason = "publish_capsules requires bool and a positive configured rate"; return result;
        }
      } else if (field == "gain_scalar") {
        if (parameter.get_type() != rclcpp::ParameterType::PARAMETER_DOUBLE ||
          !std::isfinite(parameter.as_double()) || parameter.as_double() < 0) {
          result.successful = false; result.reason = "gain_scalar must be finite and nonnegative"; return result;
        }
      } else if (field != "enabled" && field != "gain") {
        result.successful = false; result.reason = parameter.get_name() + " requires reloading the controller"; return result;
      }
    }
  }
  // Apply only after every collision parameter in this transaction has been validated.
  if (apply_visualization) {
    for (const auto & task : tasks) {
      const auto collision = std::dynamic_pointer_cast<SelfCollisionAvoidanceTask>(task);
      if (!collision) {continue;}
      for (const auto & parameter : parameters) {
        if (parameter.get_name() == "tasks." + task->id() + ".publish_capsules") {
          collision->set_publish_capsules(parameter.as_bool());
        }
      }
    }
  }
  return result;
}
}  // namespace task_priority_kinematic_control
