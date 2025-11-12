#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import customtkinter as ctk
import math, time, threading
import numpy as np
import sympy as sp

JOINT_COUNT = 6  # GUI sliders (first 5 used for FK)

# -------------------------
# Symbolic FK  (from Milestone 2)
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

def forward_kinematics_position_from_first5(q_vals):
    q = [float(q_vals[i]) for i in range(5)]
    T = np.array(f_T(*q), dtype=float)
    return T[:3, 3]

# -------------------------
# Velocity & Acceleration Kinematics (J, Jdot)
# -------------------------
pos_sym = sp.Matrix([T05[0,3], T05[1,3], T05[2,3]])
q_syms = sp.Matrix([q1, q2, q3, q4, q5])
J_sym = pos_sym.jacobian(q_syms)
dq_syms = sp.symbols('dq1 dq2 dq3 dq4 dq5')
dq_syms = sp.Matrix(dq_syms)
Jdot_sym = sp.zeros(*J_sym.shape)
for i in range(len(q_syms)):
    Jdot_sym += sp.diff(J_sym, q_syms[i]) * dq_syms[i]

f_J = sp.lambdify((q1,q2,q3,q4,q5), J_sym, 'numpy')
f_Jdot = sp.lambdify(
    (q1,q2,q3,q4,q5,dq_syms[0],dq_syms[1],dq_syms[2],dq_syms[3],dq_syms[4]),
    Jdot_sym,'numpy'
)

# -------------------------
# ROS2 Node + GUI
# -------------------------
class ArmMoverGUI(Node):
    def __init__(self):
        super().__init__('arm_mover_ms3')
        self.pub_joints = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.pub_ee = self.create_publisher(Float64MultiArray, '/ee_position', 10)
        self.pub_ee_vel = self.create_publisher(Float64MultiArray, '/ee_vel_acc', 10)

        # State
        self.start_time = time.time()
        self.wave_active = False
        self.sliders = None
        self.value_labels = None
        self.gui_ready = False

        # For velocity/accel
        self.prev_q = np.zeros(JOINT_COUNT)
        self.prev_dq = np.zeros(JOINT_COUNT)
        self.prev_time = time.time()

        # Optional: subscribe to simulator EE pose
        self.use_sim_ee_topic = True
        try:
            from geometry_msgs.msg import PoseStamped
            self.sub_sim_ee = self.create_subscription(
                PoseStamped, '/sim/ee_pose', self._sim_ee_cb, 10
            )
            self.latest_sim_ee = None
        except Exception as e:
            self.get_logger().warn(f"Sim EE topic unavailable: {e}")
            self.use_sim_ee_topic = False

        # Logging
        self.log_path = "ms3_log.csv"
        with open(self.log_path, "w") as f:
            header = "t," + ",".join([f"q{i+1}" for i in range(JOINT_COUNT)]) + "," \
                     + ",".join([f"dq{i+1}" for i in range(JOINT_COUNT)]) + "," \
                     + ",".join([f"ddq{i+1}" for i in range(JOINT_COUNT)]) + "," \
                     + "ee_px,ee_py,ee_pz,ee_vx,ee_vy,ee_vz,ee_ax,ee_ay,ee_az"
            if self.use_sim_ee_topic:
                header += ",sim_px,sim_py,sim_pz"
            f.write(header + "\n")

        # GUI thread
        threading.Thread(target=self.build_gui, daemon=True).start()
        self.create_timer(0.05, self.safe_publish_command)
        self.get_logger().info("✅ ArmMoverGUI node for Milestone 3 started.")

    # ---------- ROS2 ----------
    def _sim_ee_cb(self, msg):
        p = msg.pose.position
        self.latest_sim_ee = np.array([p.x, p.y, p.z])

    def safe_publish_command(self):
        if self.gui_ready:
            self.publish_command()

    # ---------- Main publish cycle ----------
    def publish_command(self):
        msg = Float64MultiArray()
        if self.wave_active:
            t = time.time() - self.start_time
            values = [
                0.8 * math.sin(t),
                0.6 * math.sin(0.7 * t),
                0.4 * math.cos(0.5 * t),
                0.3 * math.sin(t / 2),
                -0.4 * math.sin(0.3 * t),
                0.2 * math.cos(0.4 * t),
            ]
            for i in range(JOINT_COUNT):
                self.sliders[i].set(values[i])
                self.value_labels[i].configure(text=f"{values[i]:.2f}")
            msg.data = values
        else:
            msg.data = [self.sliders[i].get() for i in range(JOINT_COUNT)]

        self.pub_joints.publish(msg)

        # --- Compute numeric dq, ddq ---
        now = time.time()
        dt = max(1e-5, now - self.prev_time)
        q_curr = np.array(msg.data[:5])
        dq = (q_curr - self.prev_q[:5]) / dt
        ddq = (dq - self.prev_dq[:5]) / dt

        # --- Analytic velocity & acceleration ---
        try:
            J = np.array(f_J(*q_curr), dtype=float)
            Jdot = np.array(f_Jdot(*q_curr, *dq), dtype=float)
            dq_v = dq.reshape((5,1))
            ddq_v = ddq.reshape((5,1))
            xdot = (J @ dq_v).flatten()
            xddot = (J @ ddq_v + Jdot @ dq_v).flatten()
        except Exception as e:
            self.get_logger().error(f"Kinematic eval error: {e}")
            xdot = np.zeros(3)
            xddot = np.zeros(3)

        # --- FK EE position ---
        try:
            ee_pos = forward_kinematics_position_from_first5(q_curr)
            out = Float64MultiArray()
            out.data = ee_pos.tolist()
            self.pub_ee.publish(out)
        except Exception as e:
            ee_pos = np.zeros(3)
            self.get_logger().error(f"FK error: {e}")

        # --- Publish EE velocity & acceleration ---
        velmsg = Float64MultiArray()
        velmsg.data = [*xdot, *xddot]
        self.pub_ee_vel.publish(velmsg)

        # --- Log everything ---
        sim_xyz = [None, None, None]
        if self.use_sim_ee_topic and self.latest_sim_ee is not None:
            sim_xyz = self.latest_sim_ee
        try:
            with open(self.log_path, "a") as f:
                row = f"{now}," \
                      + ",".join([f"{v:.6f}" for v in q_curr]) + "," \
                      + ",".join([f"{v:.6f}" for v in dq]) + "," \
                      + ",".join([f"{v:.6f}" for v in ddq]) + "," \
                      + f"{ee_pos[0]:.6f},{ee_pos[1]:.6f},{ee_pos[2]:.6f}," \
                      + f"{xdot[0]:.6f},{xdot[1]:.6f},{xdot[2]:.6f}," \
                      + f"{xddot[0]:.6f},{xddot[1]:.6f},{xddot[2]:.6f}"
                if sim_xyz[0] is not None:
                    row += "," + ",".join([f"{v:.6f}" for v in sim_xyz])
                f.write(row + "\n")
        except Exception as e:
            self.get_logger().error(f"Log write error: {e}")

        self.prev_q[:5] = q_curr
        self.prev_dq[:5] = dq
        self.prev_time = now

        if not hasattr(self, "_last_log") or now - self._last_log > 1.0:
            self.get_logger().info(
                f"EE pos:{ee_pos.round(3)} vel:{xdot.round(3)} acc:{xddot.round(3)}"
            )
            self._last_log = now

    # ---------- GUI ----------
    def build_gui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.app = ctk.CTk()
        self.app.title("🤖 6-DOF Arm Controller — Milestone 3")
        self.app.geometry("720x560")

        title = ctk.CTkLabel(self.app, text="6-DOF Robotic Arm Controller (M3)",
                             font=("Segoe UI", 22, "bold"), text_color="#00FFB3")
        title.pack(pady=(12, 8))

        self.sliders, self.value_labels = [], []
        frame = ctk.CTkFrame(self.app, fg_color="#1E1E1E", corner_radius=10)
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        for i in range(JOINT_COUNT):
            row = ctk.CTkFrame(frame, fg_color="#1E1E1E")
            row.pack(fill="x", pady=6, padx=10)
            lbl = ctk.CTkLabel(row, text=f"Joint {i+1}", width=80, font=("Segoe UI", 14))
            lbl.pack(side="left", padx=8)
            slider = ctk.CTkSlider(row, from_=-3.14, to=3.14, width=420,
                                   progress_color="#00FFB3", button_color="#007A5E")
            slider.set(0.0)
            slider.pack(side="left", padx=(10,6))
            val_label = ctk.CTkLabel(row, text="0.00", width=60, font=("Consolas", 13))
            val_label.pack(side="left")
            self.sliders.append(slider)
            self.value_labels.append(val_label)
            slider.configure(command=lambda v, i=i: val_label.configure(text=f"{float(v):.2f}"))

        btn_frame = ctk.CTkFrame(self.app, fg_color="#111111")
        btn_frame.pack(fill="x", pady=12, padx=18)
        ctk.CTkButton(btn_frame, text="Send Once",
                      fg_color="#007A5E", command=self.send_once).pack(side="left", padx=14)
        self.wave_btn = ctk.CTkButton(btn_frame, text="Wave Motion 🌊",
                                      fg_color="#2E2E2E", command=self.toggle_wave)
        self.wave_btn.pack(side="left", padx=8)
        ctk.CTkButton(btn_frame, text="Quit",
                      fg_color="#8B0000", command=self.close_app).pack(side="right", padx=8)

        footer = ctk.CTkLabel(self.app,
            text="Publishes /joint_commands, /ee_position, /ee_vel_acc",
            font=("Segoe UI",11), text_color="#777777")
        footer.pack(pady=(6,10))

        self.gui_ready = True
        self.app.mainloop()

    # ---------- Button callbacks ----------
    def send_once(self):
        msg = Float64MultiArray()
        msg.data = [self.sliders[i].get() for i in range(JOINT_COUNT)]
        self.pub_joints.publish(msg)
        ee_pos = forward_kinematics_position_from_first5(msg.data)
        out = Float64MultiArray(); out.data = ee_pos.tolist()
        self.pub_ee.publish(out)
        self.get_logger().info(f"Sent q and EE pos {out.data}")

    def toggle_wave(self):
        self.wave_active = not self.wave_active
        if self.wave_active:
            self.start_time = time.time()
            self.wave_btn.configure(text="Stop Wave ❌", fg_color="#007A5E")
        else:
            self.wave_btn.configure(text="Wave Motion 🌊", fg_color="#2E2E2E")

    def close_app(self):
        self.get_logger().info("Closing GUI…")
        try: self.app.destroy()
        except: pass
        self.destroy_node()

# -------------------------
def main():
    rclpy.init()
    node = ArmMoverGUI()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()