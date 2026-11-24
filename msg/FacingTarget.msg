# Target for task_priority_kinematic_control/EndEffectorsFacingTask.
# Both end effectors face each other along the line that joins them.

std_msgs/Header header

# Axis of the left EE (expressed in its own frame) that points to the right EE.
# Must be a pure signed axis: +-x, +-y or +-z (e.g. {z: 1.0}).
geometry_msgs/Vector3 left_pointing_axis

# Axis of the right EE (expressed in its own frame) that points to the left EE.
# Must be a pure signed axis: +-x, +-y or +-z.
geometry_msgs/Vector3 right_pointing_axis

# Relative roll around the line left EE -> right EE [rad].
# Zero reference axis of each EE (cyclic, same sign as its pointing axis):
#   pointing +-x -> reference +-y, pointing +-y -> reference +-z, pointing +-z -> reference +-x.
# roll_angle is the signed angle (around the left->right line) from the left reference axis to
# the right one. Example: both pointing +z -> 0 = x axes parallel, pi = x axes opposite.
# NaN keeps the currently measured roll.
float64 roll_angle

# Distance between both end effectors [m]. Must be > 0.
float64 distance
