#include "task_priority_kinematic_control/core/stpik_check_io.hpp"

namespace task_priority_kinematic_control
{
bool StpikCheckIO::same_request(const StpikAction::Goal & a, const StpikAction::Goal & b)
{
  return a.trajectory == b.trajectory && a.task_id == b.task_id && a.tip_frame == b.tip_frame &&
    a.tolerance_orientation == b.tolerance_orientation && a.path_tolerance == b.path_tolerance &&
    a.goal_tolerance == b.goal_tolerance && a.goal_time_tolerance == b.goal_time_tolerance;
}

void StpikCheckIO::accept(const std::shared_ptr<Handle> & handle)
{
  std::scoped_lock lock(mutex_);
  handle_ = handle; cancel_.store(handle->is_canceling());
  StpikAction::Result result;
  result.outcome = StpikAction::Result::UNAVAILABLE;
  try {
    request_ = *handle->get_goal(); settings_ = options_();
    if (request_.validation_id != 0) {
      std::string reason;
      if (request_.validation_id != validated_id_ || !validated_snapshot_ ||
        !same_request(request_, validated_request_) ||
        settings_.configuration_key() != validated_settings_.configuration_key() ||
        !validate_(*validated_snapshot_, validated_settings_, reason))
      {
        result.code = "STALE_SNAPSHOT";
        result.message = reason.empty() ? "Validation token does not match this request" : reason;
      } else {
        result.outcome = StpikAction::Result::SUCCESS; result.code = "SUCCESS";
        result.message = "STPIK snapshot remains valid immediately before execution";
        result.validation_id = validated_id_;
      }
      complete(std::move(result)); return;
    }
    validated_snapshot_.reset(); validated_id_ = 0;
    auto snapshot = capture_();
    // Metadata is immutable; the worker receives owning copies of all mutable tasks/backend.
    snapshot_ = snapshot;
    progress_time_.store(0.0); progress_total_.store(1.0);
    const auto goal = request_;
    const auto settings = settings_;
    worker_ = std::async(std::launch::async,
      [this, snapshot = std::move(snapshot), goal, settings]() mutable {
        return StpikCheck::run(std::move(snapshot), goal, settings,
          [this]() {return cancel_.load();}, [this](double time, double total) {
            progress_time_.store(time); progress_total_.store(total);
          });
      });
  } catch (const std::exception & exception) {
    result.code = "CHECK_UNAVAILABLE"; result.message = exception.what(); complete(std::move(result));
  }
}

void StpikCheckIO::complete(StpikAction::Result result)
{
  if (!handle_) {busy_.store(false); return;}
  auto response = std::make_shared<StpikAction::Result>(std::move(result));
  if (response->outcome == StpikAction::Result::FAIL && !handle_->is_canceling()) {
    auto feedback = std::make_shared<StpikAction::Feedback>();
    feedback->time_from_start = response->time_from_start;
    const double total = request_.trajectory.duration + request_.goal_time_tolerance;
    feedback->progress = total > 0.0 ? std::clamp(response->time_from_start / total, 0.0, 1.0) : 1.0;
    feedback->message = response->code + ": " + response->message;
    handle_->publish_feedback(feedback);
  }
  if (handle_->is_canceling()) {
    response->outcome = StpikAction::Result::CANCELED;
    response->code = "CANCELED"; handle_->canceled(response);
  } else if (response->outcome == StpikAction::Result::SUCCESS) {handle_->succeed(response);}
  else {handle_->abort(response);}
  handle_.reset(); snapshot_.reset(); busy_.store(false);
}

void StpikCheckIO::poll()
{
  std::scoped_lock lock(mutex_);
  if (!handle_ || !worker_.valid()) {return;}
  if (worker_.wait_for(std::chrono::milliseconds(0)) != std::future_status::ready) {
    auto feedback = std::make_shared<StpikAction::Feedback>();
    feedback->time_from_start = progress_time_.load();
    const double total = progress_total_.load();
    feedback->progress = total > 0.0 ? std::clamp(feedback->time_from_start / total, 0.0, 1.0) : 1.0;
    feedback->message = "Simulating isolated Task Priority hierarchy";
    handle_->publish_feedback(feedback); return;
  }
  StpikAction::Result result;
  try {
    result = worker_.get();
    if (cancel_.load()) {
      result.outcome = StpikAction::Result::CANCELED;
      result.code = "CANCELED"; result.message = "STPIK canceled before admission to execution";
    } else if (result.outcome == StpikAction::Result::SUCCESS) {
      std::string reason;
      if (!snapshot_ || options_().configuration_key() != settings_.configuration_key() ||
        !validate_(*snapshot_, settings_, reason)) {
        result.outcome = StpikAction::Result::UNAVAILABLE;
        result.code = "STALE_SNAPSHOT"; result.message = reason.empty() ? "STPIK settings changed during simulation" : reason;
      } else {
        validated_snapshot_ = std::move(snapshot_); validated_request_ = request_;
        validated_settings_ = settings_; validated_id_ = ++next_id_;
        result.validation_id = validated_id_;
      }
    }
  } catch (const std::exception & exception) {
    result.outcome = StpikAction::Result::UNAVAILABLE;
    result.code = "CHECK_UNAVAILABLE"; result.message = exception.what();
  }
  complete(std::move(result));
}

void StpikCheckIO::stop()
{
  cancel_.store(true);
  timer_.reset(); server_.reset();
  std::scoped_lock lock(mutex_);
  if (worker_.valid()) {
    worker_.wait();
    try {worker_.get();} catch (const std::exception &) {}
  }
  if (handle_) {
    StpikAction::Result result; result.outcome = StpikAction::Result::UNAVAILABLE;
    result.code = "CHECK_UNAVAILABLE"; result.message = "STPIK host stopped";
    complete(std::move(result));
  }
  snapshot_.reset(); validated_snapshot_.reset(); validated_id_ = 0; busy_.store(false);
}
}  // namespace task_priority_kinematic_control
