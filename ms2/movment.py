#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
import customtkinter as ctk
import math, time, threading
import numpy as np
import sympy as sp

JOINT_COUNT = 6  # number of GUI sliders (we use first 5 for FK)

# -------------------------
# Symbolic FK (T0^5) from your derivation
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
    """
    q_vals: iterable with >=5 elements (radians)
    returns np.array([x,y,z])
    """
    q = [float(q_vals[i]) for i in range(5)]
    T = np.array(f_T(*q), dtype=float)   # 4x4
    return T[:3, 3]

# -------------------------
# ROS2 Node + GUI -> uses slider values as q
# -------------------------
class ArmMoverGUI(Node):
    def __init__(self):
        super().__init__('arm_mover_modern_gui')
        self.pub_joints = self.create_publisher(Float64MultiArray, '/joint_commands', 10)
        self.pub_ee = self.create_publisher(Float64MultiArray, '/ee_position', 10)

        self.start_time = time.time()
        self.wave_active = False

        # GUI variables (set later in build_gui)
        self.sliders = None
        self.value_labels = None
        self.gui_ready = False

        # Start GUI thread (daemon so process exits cleanly)
        t = threading.Thread(target=self.build_gui, daemon=True)
        t.start()

        # Timer publishes at 20 Hz but safety-checks gui_ready
        self.create_timer(0.05, self.safe_publish_command)

        self.get_logger().info("ArmMoverGUI node started. GUI thread launching...")

    # -------- safe timer wrapper --------
    def safe_publish_command(self):
        if not self.gui_ready:
            return
        self.publish_command()

    # -------- publishing logic (uses slider values as q) --------
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
            # update sliders & labels visually
            for i in range(JOINT_COUNT):
                # ensure slider exists (it does because gui_ready True)
                self.sliders[i].set(values[i])
                self.value_labels[i].configure(text=f"{values[i]:.2f}")
            msg.data = values
        else:
            msg.data = [self.sliders[i].get() for i in range(JOINT_COUNT)]

        # publish joint commands
        self.pub_joints.publish(msg)

        # compute FK (first 5 sliders) and publish EE position
        try:
            ee_pos = forward_kinematics_position_from_first5(msg.data)
            out = Float64MultiArray()
            out.data = [float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])]
            self.pub_ee.publish(out)
            self.get_logger().info_throttle = getattr(self.get_logger(), "info", print)  # fallback
            # throttle manual: log every 1s
            now = time.time()
            if not hasattr(self, "_last_log") or now - self._last_log > 1.0:
                self.get_logger().info(f"EE pos -> x: {ee_pos[0]:.4f}, y: {ee_pos[1]:.4f}, z: {ee_pos[2]:.4f}")
                self._last_log = now
        except Exception as e:
            self.get_logger().error(f"FK eval error: {e}")

    # -------- GUI builder (customtkinter) --------
    def build_gui(self):
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("green")

        self.app = ctk.CTk()
        self.app.title("🤖 6-DOF Arm Controller (FK from first 5 q)")
        self.app.geometry("700x560")

        title = ctk.CTkLabel(self.app, text="6-DOF Robotic Arm Controller", font=("Segoe UI", 22, "bold"), text_color="#00FFB3")
        title.pack(pady=(12, 8))

        self.sliders = []
        self.value_labels = []

        frame = ctk.CTkFrame(self.app, fg_color="#1E1E1E", corner_radius=10)
        frame.pack(fill="both", expand=True, padx=20, pady=10)

        for i in range(JOINT_COUNT):
            row_frame = ctk.CTkFrame(frame, fg_color="#1E1E1E")
            row_frame.pack(fill="x", pady=6, padx=10)

            lbl = ctk.CTkLabel(row_frame, text=f"Joint {i+1}", font=("Segoe UI", 14), width=80, anchor="w")
            lbl.pack(side="left", padx=8)

            slider = ctk.CTkSlider(row_frame, from_=-3.14, to=3.14, width=420,
                                   progress_color="#00FFB3", button_color="#007A5E",
                                   button_hover_color="#00CC99")
            slider.set(0.0)
            slider.pack(side="left", padx=(10, 6))
            self.sliders.append(slider)

            val_label = ctk.CTkLabel(row_frame, text="0.00", width=60, font=("Consolas", 13), text_color="#AAAAAA")
            val_label.pack(side="left")
            self.value_labels.append(val_label)

            # update label when slider moves
            def make_update(i):
                return lambda val: self.value_labels[i].configure(text=f"{float(val):.2f}")
            slider.configure(command=make_update(i))

        # Buttons
        btn_frame = ctk.CTkFrame(self.app, fg_color="#111111")
        btn_frame.pack(fill="x", pady=12, padx=18)

        send_btn = ctk.CTkButton(btn_frame, text="Send Once", fg_color="#007A5E", hover_color="#00CC99", command=self.send_once)
        send_btn.pack(side="left", padx=14, pady=8)

        self.wave_btn = ctk.CTkButton(btn_frame, text="Wave Motion 🌊", fg_color="#2E2E2E", hover_color="#007A5E", command=self.toggle_wave)
        self.wave_btn.pack(side="left", padx=8)

        quit_btn = ctk.CTkButton(btn_frame, text="Quit", fg_color="#8B0000", hover_color="#CC0000", command=self.close_app)
        quit_btn.pack(side="right", padx=8)

        footer = ctk.CTkLabel(self.app, text="Publishing /joint_commands and /ee_position (EE from first 5 q)", font=("Segoe UI", 11), text_color="#777777")
        footer.pack(pady=(6, 10))

        # mark GUI ready and start mainloop (keeps this thread alive)
        self.gui_ready = True
        self.app.mainloop()

    # -------- Button callbacks --------
    def send_once(self):
        if not self.gui_ready:
            return
        msg = Float64MultiArray()
        msg.data = [self.sliders[i].get() for i in range(JOINT_COUNT)]
        self.pub_joints.publish(msg)
        # compute & publish EE pos immediately
        try:
            ee_pos = forward_kinematics_position_from_first5(msg.data)
            out = Float64MultiArray(); out.data = [float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])]
            self.pub_ee.publish(out)
            self.get_logger().info(f"Sent joint values and EE pos: {out.data}")
        except Exception as e:
            self.get_logger().error(f"FK eval error on send_once: {e}")

    def toggle_wave(self):
        self.wave_active = not self.wave_active
        if self.wave_active:
            self.start_time = time.time()
            self.wave_btn.configure(text="Stop Wave ❌", fg_color="#007A5E")
        else:
            self.wave_btn.configure(text="Wave Motion 🌊", fg_color="#2E2E2E")

    def close_app(self):
        self.get_logger().info("Closing GUI...")
        try:
            self.app.destroy()
        except Exception:
            pass
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
        try:
            node.destroy_node()
        except Exception:
            pass
        rclpy.shutdown()

if __name__ == '__main__':
    main()