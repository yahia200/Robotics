#!/usr/bin/env python3
"""
MCTR911 - Milestone 4
- Joint-space trajectory (quintic polynomial) between two joint configs
- Task-space trajectory (circular path) using resolved-rate IK with J(q)

Publishes:
  /joint_commands (Float64MultiArray)  -> to MuJoCo / your arm controller
  /ee_position   (Float64MultiArray)   -> actual EE position from FK
  /ee_desired    (Float64MultiArray)   -> desired EE pos (task-space mode)

Uses the same FK and Jacobian as Milestone 3.
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

import numpy as np
import sympy as sp
import math
import time

JOINT_COUNT = 6            # your robot (first 5 used for FK/Jacobian)
CONTROL_RATE = 100.0       # Hz
DT = 1.0 / CONTROL_RATE

# -------------------------
# Symbolic FK (from Milestone 2 / 3)
# -------------------------
q1, q2, q3, q4, q5 = sp.symbols('q1 q2 q3 q4 q5')

T05 = sp.Matrix([
 [sp.sin(q1)*sp.sin(q5) + sp.cos(q1)*sp.cos(q5)*sp.cos(q2+q3+q4),
  sp.cos(q1)*sp.sin(q2+q3+q4),
 -sp.sin(q1)*sp.cos(q5) + sp.sin(q5)*sp.cos(q1)*sp.cos(q2+q3+q4),
 (0.2*sp.sin(q2+q3+q4) + sp.cos(q2) + sp.cos(q2+q3))*sp.cos(q1)],

 [sp.sin(q1)*sp.cos(q5)*sp.cos(q2+q3+q4) - sp.sin(q5)*sp.cos(q1),
  sp.sin(q1)*sp.sin(q2+q3+q4),
  sp.sin(q1)*sp.sin(q5)*sp.cos(q2+q3+q4) + sp.cos(q1)*sp.cos(q5),
 (0.2*sp.sin(q2+q3+q4) + sp.cos(q2) + sp.cos(q2+q3))*sp.sin(q1)],

 [sp.sin(q2+q3+q4)*sp.cos(q5),
 -sp.cos(q2+q3+q4),
 sp.sin(q5)*sp.sin(q2+q3+q4),
 sp.sin(q2) + sp.sin(q2+q3) - 0.2*sp.cos(q2+q3+q4) + 0.2],

 [0, 0, 0, 1]
])

f_T = sp.lambdify((q1, q2, q3, q4, q5), T05, 'numpy')

def fk_pos(q_vals_5):
    """End-effector position from first 5 joints."""
    T = np.array(f_T(*q_vals_5), dtype=float)
    return T[:3, 3]

# -------------------------
# Jacobian (from Milestone 3)
# -------------------------
pos_sym = sp.Matrix([T05[0,3], T05[1,3], T05[2,3]])
q_syms = sp.Matrix([q1, q2, q3, q4, q5])
J_sym = pos_sym.jacobian(q_syms)

f_J = sp.lambdify((q1, q2, q3, q4, q5), J_sym, 'numpy')

# -------------------------
# Quintic Polynomial (Joint-space trajectory)
# -------------------------
def quintic_coeffs(q0, qf, T):
    """
    Returns a0..a5 arrays for each joint for a quintic that satisfies:
      q(0) = q0, q(T) = qf
      dq(0) = dq(T) = 0
      ddq(0) = ddq(T) = 0
    """
    q0 = np.array(q0, dtype=float)
    qf = np.array(qf, dtype=float)
    dq0 = np.zeros_like(q0)
    dqf = np.zeros_like(qf)
    ddq0 = np.zeros_like(q0)
    ddqf = np.zeros_like(qf)

    T2 = T**2
    T3 = T**3
    T4 = T**4
    T5 = T**5

    a0 = q0
    a1 = dq0
    a2 = ddq0 / 2.0
    a3 = ( 20*(qf - q0) - (8*dqf + 12*dq0)*T - (3*ddq0 - ddqf)*T2 ) / (2*T3)
    a4 = ( 30*(q0 - qf) + (14*dqf + 16*dq0)*T + (3*ddq0 - 2*ddqf)*T2 ) / (2*T4)
    a5 = ( 12*(qf - q0) - (6*dqf + 6*dq0)*T - (ddq0 - ddqf)*T2 ) / (2*T5)

    return a0, a1, a2, a3, a4, a5

def quintic_eval(a0,a1,a2,a3,a4,a5, t):
    """Return q(t), dq(t), ddq(t) at time t."""
    t2 = t*t
    t3 = t2*t
    t4 = t3*t
    t5 = t4*t

    q  = a0 + a1*t + a2*t2 + a3*t3 + a4*t4 + a5*t5
    dq = a1 + 2*a2*t + 3*a3*t2 + 4*a4*t3 + 5*a5*t4
    ddq = 2*a2 + 6*a3*t + 12*a4*t2 + 20*a5*t3
    return q, dq, ddq

# -------------------------
# Task-space circle definition
# -------------------------
def desired_taskspace_circle(t, radius=0.5, omega=1,
                             center=np.array([1.0, 0.0, -1.0])):
    """
    Simple circular trajectory in XY plane (Z fixed):
      p_d(t) = center + [r cos(ωt), r sin(ωt), 0]
    """
    px = center[0] + radius * math.cos(omega * t)
    py = center[1] + radius * math.sin(omega * t)
    pz = center[2]
    return np.array([px, py, pz], dtype=float)

# -------------------------
# Node
# -------------------------
class Milestone4TrajectoryNode(Node):
    def __init__(self, mode="joint_space_demo"):
        super().__init__("ms4_trajectory_node")

        self.mode = mode  # "joint_space_demo" or "task_space_circle"
        self.pub_joints = self.create_publisher(Float64MultiArray,
                                                "/joint_commands", 10)
        self.pub_ee = self.create_publisher(Float64MultiArray,
                                            "/ee_position", 10)
        self.pub_ee_des = self.create_publisher(Float64MultiArray,
                                                "/ee_desired", 10)

        self.timer = self.create_timer(DT, self._timer_cb)
        self.start_time = time.time()

        # ----- JOINT-SPACE TRAJECTORY SETUP -----
        # Only first 5 joints are relevant for EE, 6th kept constant.
        self.T_total_joint = 8.0  # seconds
        q0 = np.array([0.0,  0.0,  0.0,  0.0,  0.0], dtype=float)
        qf = np.array([ 0.8, 0.0,  0.0, 0.0,  0.0], dtype=float)
        (self.a0, self.a1, self.a2,
         self.a3, self.a4, self.a5) = quintic_coeffs(q0, qf,
                                                     self.T_total_joint)

        # ----- TASK-SPACE RESOLVED-RATE INITIALIZATION -----
        self.q_task = np.array([0.3, 0.0, 0.0, 0.0, 0.0], dtype=float)
        self.q6_const = 0.0
        self.k_p = 2.0   # proportional gain in task space
        self.last_t = 0.0

        # simple CSV logging
        self.log_path = "ms4_trajectories_log.csv"
        with open(self.log_path, "w") as f:
            f.write("t,mode," +
                    ",".join([f"q{i+1}" for i in range(JOINT_COUNT)]) + "," +
                    "ee_x,ee_y,ee_z,des_x,des_y,des_z\n")

        self.get_logger().info(f"✅ Milestone 4 node started in mode: {self.mode}")

    # -------------------------
    def _timer_cb(self):
        t = time.time() - self.start_time

        if self.mode == "joint_space_demo":
            self._step_joint_space(t)
        elif self.mode == "task_space_circle":
            self._step_task_space(t)
        else:
            self.get_logger().warn(f"Unknown mode {self.mode}")

    # -------------------------
    # 1) Joint-space quintic demo
    # -------------------------
    def _step_joint_space(self, t):
        # hold final position after motion is finished
        if t > self.T_total_joint:
            t_eval = self.T_total_joint
        else:
            t_eval = t

        q5, dq5, ddq5 = quintic_eval(self.a0,self.a1,self.a2,
                                     self.a3,self.a4,self.a5, t_eval)
        # append joint 6 constant = 0
        q_cmd = np.concatenate([q5, np.array([0.0])])

        # publish joint commands
        msg = Float64MultiArray()
        msg.data = q_cmd.tolist()
        self.pub_joints.publish(msg)

        # compute EE via FK
        ee = fk_pos(q5)
        ee_msg = Float64MultiArray(); ee_msg.data = ee.tolist()
        self.pub_ee.publish(ee_msg)

        # no specific desired EE here, use actual for logging consistency
        des = ee.copy()
        des_msg = Float64MultiArray(); des_msg.data = des.tolist()
        self.pub_ee_des.publish(des_msg)

        self._log_row(t, "joint", q_cmd, ee, des)

    # -------------------------
    # 2) Task-space circular trajectory with resolved-rate IK
    # -------------------------
    def _step_task_space(self, t):
        # Desired EE position on circle
        p_des = desired_taskspace_circle(t)

        # Current EE position and Jacobian
        try:
            p_curr = fk_pos(self.q_task)
            J = np.array(f_J(*self.q_task), dtype=float)  # 3x5
        except Exception as e:
            self.get_logger().error(f"Kinematics error: {e}")
            return

        # Position error
        e = p_des - p_curr

        # Desired EE velocity (simple P controller in task space)
        v_des = self.k_p * e

        # map to joint velocities via damped pseudoinverse
        lam = 1e-3
        JT = J.T
        JJT = J @ JT
        dq = JT @ np.linalg.inv(JJT + lam * np.eye(3)) @ v_des  # 5-dim

        # integrate to get new q
        self.q_task = self.q_task + dq * DT

        # combine with joint 6 constant
        q_cmd = np.concatenate([self.q_task, np.array([self.q6_const])])

        # publish joint commands
        msg = Float64MultiArray(); msg.data = q_cmd.tolist()
        self.pub_joints.publish(msg)

        # publish actual EE
        p_curr_new = fk_pos(self.q_task)
        ee_msg = Float64MultiArray(); ee_msg.data = p_curr_new.tolist()
        self.pub_ee.publish(ee_msg)

        # publish desired EE for comparison
        des_msg = Float64MultiArray(); des_msg.data = p_des.tolist()
        self.pub_ee_des.publish(des_msg)

        self._log_row(t, "task", q_cmd, p_curr_new, p_des)

    # -------------------------
    def _log_row(self, t, mode, q_cmd, ee, des):
        try:
            with open(self.log_path, "a") as f:
                f.write(f"{t:.4f},{mode}," +
                        ",".join([f"{v:.6f}" for v in q_cmd]) + "," +
                        f"{ee[0]:.6f},{ee[1]:.6f},{ee[2]:.6f}," +
                        f"{des[0]:.6f},{des[1]:.6f},{des[2]:.6f}\n")
        except Exception as e:
            self.get_logger().error(f"Log error: {e}")

# -------------------------
def main():
    rclpy.init()
    # change mode here if you want "task_space_circle"
    # node = Milestone4TrajectoryNode(mode="task_space_circle")
    node = Milestone4TrajectoryNode(mode="joint_space_demo")

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()