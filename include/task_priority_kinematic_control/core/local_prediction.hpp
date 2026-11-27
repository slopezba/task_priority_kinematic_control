#pragma once
#include "task_priority_kinematic_control/core/stpik_check.hpp"

namespace task_priority_kinematic_control
{
StpikAction::Result predict_local_segment(
  const StpikSnapshot &, const StpikAction::Goal &,
  const StpikOptions &, const StpikCheck::Canceled & canceled = {});
}
