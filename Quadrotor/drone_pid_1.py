import numpy as np
import math
import matplotlib.pyplot as plt


class Position_PID():
	def __init__(self, Kp, dt):

		####################################################################
		# Initialize self.pos_gain_p as an np array. 
		# Kp (x, y, z) is the partial gain for the position PID controller
		####################################################################
		self.pos_gain_p = np.array(Kp)

		########################################################################
		# End TODO
		########################################################################

		self.vel_int = 0
		self.g = 9.8
		self.dt = dt
		self.velocity_integral_horizontal = np.array([0.0, 0.0])
		self.velocity_integral_vertical = 0.0

	def set_velocity_gain(self, Kp, Ki, Kd):
		####################################################################
		# Initialize self.vel_gain_p as an np array. 
		# Initialize self.vel_gain_i as an np array. 
		# Initialize self.vel_gain_d as an np array. 
		# Kp (x, y, z), Ki (x, y, z), Kd (x, y, z) are the partial gain, 
		# integral gain, and derivative gain respectively 
		# for the velocity PID controller
		####################################################################
		self.vel_gain_p = np.array(Kp)
		self.vel_gain_i = np.array(Ki)
		self.vel_gain_d = np.array(Kd)

		########################################################################
		# End TODO
		########################################################################

	def set_velocity_limit(self, vel_horizontal, vel_up, vel_down):
		####################################################################
		# Initialize horizontal velocity limit self.vel_horizontal_lim
		# Initialize up velocity limit self.vel_up_lim 
		# Initialize down velocity limit self.vel_down_lim 
		####################################################################
		self.vel_horizontal_lim = vel_horizontal
		self.vel_up_lim = vel_up
		self.vel_down_lim = vel_down

		########################################################################
		# End TODO
		########################################################################

	def set_state(self, position, velocity, yaw, acceleration):
		####################################################################
		# Set the current states of the UAV as numpy arrays
		# self.pos is the current position of the drone with 3 dimensions
		# self.vel is the current velocity of the drone with 3 dimensions
		# self.yaw is the current yaw of the dronne with 1 dimension
		# self.vel_dot is acceleration of the drone with 3 dimensions
		####################################################################
		self.pos = np.array(position)
		self.vel = np.array(velocity)
		self.yaw = yaw
		self.vel_dot = np.array(acceleration)

		########################################################################
		# End TODO
		########################################################################

	def position_setpoint(self, position_sp, velocity_sp=(0.0, 0.0, 0.0), yaw_sp=0.0, yawspeed_sp=0.0, acceleration_sp=(0.0, 0.0, 0.0)):
		####################################################################
		# Set the position setpoint of the UAV as numpy arrays
		# self.pos_sp is the position setpoint of the drone with 3 dimensions
		# You can ignore the other setpoints, 
		# we are not using them in this assignment
		####################################################################
		self.pos_sp = np.array(position_sp)

		########################################################################
		# End TODO
		########################################################################

		self.vel_sp = np.array(velocity_sp)
		self.acc_sp = np.array(acceleration_sp)
		self.yaw_sp = yaw_sp
		self.yawspeed_sp = yawspeed_sp

	def position_control(self):
		####################################################################
		# velocity_setpoint = Kp(p_setpoint - p) (Refer to Flight control slide page 31)
		# Calculate desired speed (velocity_setpoint) using above equation
		####################################################################
		vel_sp_position = self.pos_gain_p * (self.pos_sp - self.pos)

		########################################################################
		# End TODO
		########################################################################

		self.vel_sp[:2] = clip(vel_sp_position[:2], -self.vel_horizontal_lim, self.vel_horizontal_lim)
		self.vel_sp[2] = clip(vel_sp_position[2], -self.vel_down_lim, self.vel_up_lim)

	def velocity_control(self):
		####################################################################
		# Theta_setpoint = 1/g * A_yaw(Kp * velocity_error_horizontal + Ki * self.velocity_integral_horizontal + Kd * self.vel_dot_horizontal)
		# (Refer to Flight control slide page 31)
		# Calculate desired theta using the equation above
		# accelerate_vertical = g + kp * velocity_error_vertical + ki * self.velocity_integral_vertical + kd * self.vel_dot_vertical
		# You can ignore the other setpoints, 
		# we are not using them in this assignment
		####################################################################
		velocity_error = self.vel_sp - self.vel
		self.velocity_integral_horizontal += velocity_error[:2] * self.dt
		vel_dot_horizontal = self.vel_dot[:2]
		control_input_horizontal = (
			self.vel_gain_p[:2] * velocity_error[:2]
			+ self.vel_gain_i[:2] * self.velocity_integral_horizontal
			+ self.vel_gain_d[:2] * vel_dot_horizontal
		)

		# Rotate control_input_horizontal from world frame to body frame
		cos_yaw = math.cos(self.yaw)
		sin_yaw = math.sin(self.yaw)

		# World to body frame transform
		u_x = cos_yaw * control_input_horizontal[0] + sin_yaw * control_input_horizontal[1]
		u_y = -sin_yaw * control_input_horizontal[0] + cos_yaw * control_input_horizontal[1]

		theta_x = u_y / self.g  # roll
		theta_y = u_x / self.g  # pitch
		self.theta_h = np.array([theta_x, theta_y])

		# vertical
		velocity_error_vertical = velocity_error[2]
		self.velocity_integral_vertical += velocity_error_vertical * self.dt
		vel_dot_vertical = self.vel_dot[2]
		self.acc_v_sp_velocity = (
			self.g
			+ self.vel_gain_p[2] * velocity_error_vertical
			+ self.vel_gain_i[2] * self.velocity_integral_vertical
			+ self.vel_gain_d[2] * vel_dot_vertical
		)

		########################################################################
		# End TODO
		########################################################################

	def calc_next_position(self, current_pose=None, dt=None):
		####################################################################
		# Do not change the code in this method
		####################################################################

		if dt != None:
			self.dt = dt
		if current_pose == None:

			#acc_x = self.acc_v_sp_velocity * math.tan(self.theta_h[1])
			acc_x = self.g * math.tan(self.theta_h[1])
			delta_x = self.vel[0] * self.dt + 0.5 * acc_x * self.dt ** 2
			delta_x_vel = acc_x * self.dt

			#acc_y = self.acc_v_sp_velocity * math.tan(self.theta_h[0])
			acc_y = self.g * math.tan(self.theta_h[0])
			delta_y = self.vel[1] * self.dt + 0.5 * acc_y * self.dt ** 2
			delta_y_vel = acc_y * self.dt

			acc_z = self.acc_v_sp_velocity - self.g
			delta_z = self.vel[2] * self.dt + 0.5 * acc_z * self.dt ** 2
			delta_z_vel = acc_z * self.dt

			self.pos += np.array((delta_x, delta_y, delta_z))
			self.vel += np.array((delta_x_vel, delta_y_vel, delta_z_vel))
			self.vel_dot = np.array((acc_x, acc_y, acc_z))

		else:
			current_pose = np.array(current_pose)
			print(current_pose)
			'''
			acc_x = self.acc_v_sp_velocity * math.tan(self.theta_h[1])
			acc_y = self.acc_v_sp_velocity * math.tan(self.theta_h[0])
			'''
			acc_x = self.g * math.tan(self.theta_h[1])
			acc_y = self.g * math.tan(self.theta_h[0])
			acc_z = self.acc_v_sp_velocity - self.g

			vel = (current_pose - self.pos) / self.dt
			self.pos = current_pose
			self.vel = vel
			self.vel_dot = np.array((acc_x, acc_y, acc_z))

		# print(self.pos)
		return self.pos, self.vel_dot


def clip(value, min_lim, max_lim):
	return np.minimum(np.maximum(value, min_lim), max_lim)


if __name__ == '__main__':

	########################################################################
	# You are required to adjust the following PID parameters
	# Kp_p is the proportional gain for position control
	# Kp_v is the proportional gain for velocity control
	# Ki_v is the integral gain for velocity control
	# Kd_v is the derivative gain for velocity control
	########################################################################
	Kp_p = (0.2, 0.2, 3.0)   # More aggressive in Z (altitude)
	Kp_v = (2.0, 2.0, 2.5)
	Ki_v = (0.1, 0.1, 0.2)   # Small values to avoid windup
	Kd_v = (0.3, 0.3, 0.4)   # Helps smoothen velocity overshoots

	########################################################################
	# End TODO
	########################################################################

	########################################################################
	# If executing takeoff task, set takeoff_task to True. Otherwise false
	########################################################################

	takeoff_task = False

	if takeoff_task:
		goal = (0.0, 0.0, 10.0)
	else:
		goal = (10.0, 10.0, 10.0)

	########################################################################
	# End TODO
	########################################################################

	t = range(300)

	posPID = Position_PID(Kp_p, 0.11)
	posPID.set_velocity_gain(Kp_v, Ki_v, Kd_v)
	posPID.set_velocity_limit(12, 12, 12)
	posPID.set_state((0.0, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0), (0.0, 0.0, 0.0))
	posPID.position_setpoint(goal)
	traj = np.zeros((len(t), 3))
	for i in t:
		posPID.position_control()
		posPID.velocity_control()
		pos, acc = posPID.calc_next_position()

		traj[i, :] = pos

	traj = np.array(traj)

	ax1 = plt.subplot(311)
	ax1.plot(t, traj[:, 0])
	ax1.set_ylabel("X Position")

	ax2 = plt.subplot(312)
	ax2.plot(t, traj[:, 1])
	ax2.set_ylabel("Y Position")

	ax3 = plt.subplot(313)
	ax3.plot(t, traj[:, 2])
	ax3.set_ylabel("Z Position")
	ax3.set_xlabel("Time Step")

	plt.xlim(xmin=0)
	plt.tight_layout()
	plt.show()
