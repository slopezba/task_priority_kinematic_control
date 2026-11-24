#pragma once
#include "task_priority_kinematic_control/core/initial_posture.hpp"
#include "task_priority_kinematic_control/tasks/pose_task_common.hpp"
#include <moveit/planning_scene/planning_scene.h>
#include <urdf_parser/urdf_parser.h>
#include <srdfdom/model.h>
#include <cmath>
#include <limits>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

namespace task_priority_kinematic_control
{
// Frozen MoveIt planning scene used by offline posture and nominal planning.
class PostureScene
{
public:
  PostureScene(const Preparation::Goal & g, const WholeBodyModel & model)
  : names_(model.all_joint_names())
  {
    auto u = urdf::parseURDF(g.robot_description);
    auto srdf = std::make_shared<srdf::Model>();
    if (!u || !srdf->initString(*u, g.robot_description_semantic)) {
      throw std::runtime_error("Invalid planning robot description");
    }
    auto robot = std::make_shared<moveit::core::RobotModel>(u, srdf);
    if (robot->getModelFrame() != model.world_frame() ||
      robot->getRootLinkName() != model.base_frame() ||
      robot->getRootJoint()->getType() != moveit::core::JointModel::FLOATING)
    {
      throw std::runtime_error("Planning scene requires a matching floating base and world");
    }
    scene_ = std::make_unique<planning_scene::PlanningScene>(robot);
    if (g.scene.is_diff || !scene_->setPlanningSceneMsg(g.scene)) {
      throw std::runtime_error("Invalid complete planning scene");
    }
    for (const auto & n:names_) {
      if (!robot->hasJointModel(n) || robot->getJointModel(n)->getVariableCount() != 1) {
        throw std::runtime_error("Unmapped planning joint: " + n);
      }
    }
  }
  double clearance(const WholeBodyState & s)
  {
    auto & state = scene_->getCurrentStateNonConst();
    state.setJointPositions(state.getRobotModel()->getRootJoint(), pose_math::base_pose_from_state(s));
    for (size_t i = 0; i < names_.size(); ++i) {
      state.setVariablePosition(names_[i], s.joint_positions(i));
    }
    state.update();
    if (scene_->isStateColliding(state)) {return 0;}
    const auto & acm = scene_->getAllowedCollisionMatrix();
    world_clearance_ = scene_->distanceToCollision(state);
    return std::min(
      world_clearance_,
      scene_->getCollisionEnv()->distanceSelf(state, acm));
  }
  // Signed minimum distance (negative = penetration depth) to self and world, so an
  // optimizer still has a gradient inside a collision. Same model state as clearance().
  double signed_clearance(const WholeBodyState & s)
  {
    auto & state = scene_->getCurrentStateNonConst();
    state.setJointPositions(state.getRobotModel()->getRootJoint(), pose_math::base_pose_from_state(s));
    for (size_t i = 0; i < names_.size(); ++i) {
      state.setVariablePosition(names_[i], s.joint_positions(i));
    }
    state.update();
    collision_detection::DistanceRequest request;
    request.enable_signed_distance = true;
    request.type = collision_detection::DistanceRequestType::GLOBAL;
    request.acm = &scene_->getAllowedCollisionMatrix();
    collision_detection::DistanceResult result;
    scene_->getCollisionEnv()->distanceSelf(request, result, state);
    double distance = result.minimum_distance.distance;
    result.clear();
    scene_->getCollisionEnv()->distanceRobot(request, result, state);
    distance = std::min(distance, result.minimum_distance.distance);
    return std::isfinite(distance) ? distance : std::numeric_limits<double>::infinity();
  }
  double world_clearance() const {return world_clearance_;}
  std::string collision_diagnostic() const
  {
    collision_detection::CollisionRequest request;
    request.contacts = true;
    request.max_contacts = 5;
    request.max_contacts_per_pair = 1;
    collision_detection::CollisionResult result;
    scene_->checkCollision(request, result, scene_->getCurrentState());
    std::ostringstream detail;
    detail << "Planning scene collision; source=MoveIt";
    if (result.contacts.empty()) {
      detail << "; contacts=unavailable (collision or nonpositive clearance reported)";
    }
    for (const auto & pair : result.contacts) {
      if (pair.second.empty()) {continue;}
      const auto & contact = pair.second.front();
      const bool world = contact.body_type_1 == collision_detection::BodyTypes::WORLD_OBJECT ||
        contact.body_type_2 == collision_detection::BodyTypes::WORLD_OBJECT;
      detail << "; contact={pair=" << pair.first.first << " <-> " << pair.first.second
             << "; type=" << (world ? "robot_world" : "self_collision")
             << "; penetration=" << contact.depth << " m}";
    }
    return detail.str();
  }

private:
  std::vector<std::string> names_;
  std::unique_ptr<planning_scene::PlanningScene> scene_;
  double world_clearance_ = std::numeric_limits<double>::infinity();
};
}  // namespace task_priority_kinematic_control
