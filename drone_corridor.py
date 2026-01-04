import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from functools import partial
from mpl_toolkits.mplot3d import Axes3D
from optimize import BatchedADMM
import time
import os
import shutil   
import io
import imageio.v2 as imageio

# =========================================================
# 1. Constants & B-Spline
# =========================================================
MASS = 0.904
G = 9.81
J_DIAG = jnp.array([0.0023, 0.0026, 0.0032])
J_INV = 1.0 / J_DIAG

class BSplineBasis:
    def __init__(self, n_cp, horizon):
        self.n_cp = n_cp
        self.horizon = horizon
        t = jnp.linspace(0, 1, horizon)
        self.basis_mat = self._generate_basis(t, n_cp)

    def _generate_basis(self, t, n_cp):
        basis = []
        for i in range(n_cp):
            center = i / (n_cp - 1)
            val = jnp.exp(-0.5 * ((t - center) / 0.15)**2)
            basis.append(val)
        basis = jnp.stack(basis, axis=1)
        basis = basis / (jnp.sum(basis, axis=1, keepdims=True) + 1e-6)
        return basis

    @partial(jax.jit, static_argnums=(0,))
    def get_sequence(self, coeffs):
        return jnp.dot(self.basis_mat, coeffs)

# =========================================================
# 2. Dynamics (Quadrotor)
# =========================================================

@jax.jit
def se3_step(state, u, dt):
    pos, vel, rpy, omega = state[0:3], state[3:6], state[6:9], state[9:12]
    thrust, torques = u[0], u[1:]
    
    thrust = jnp.clip(thrust, 0.0, 20.0)
    torques = jnp.clip(torques, -1.0, 1.0)
    
    phi, theta, psi = rpy
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)
    
    R_vec = jnp.array([
        cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi,
        spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi,
        -sth,     cth*sphi,                  cth*cphi
    ]).reshape(3,3)

    drag_lin = -0.1 * vel
    drag_ang = -0.1 * omega
    
    acc = (R_vec @ jnp.array([0., 0., thrust])) / MASS - jnp.array([0., 0., G]) + drag_lin
    omega_dot = J_INV * (torques - jnp.cross(omega, J_DIAG * omega) + drag_ang)
    
    next_pos = pos + vel * dt
    next_vel = vel + acc * dt
    next_rpy = rpy + omega * dt 
    next_omega = omega + omega_dot * dt
    
    return jnp.concatenate([next_pos, next_vel, next_rpy, next_omega])

def get_linear_model(dt):
    A = jnp.eye(12)
    A = A.at[0:3, 3:6].set(jnp.eye(3) * dt)
    A = A.at[6:9, 9:12].set(jnp.eye(3) * dt)
    A = A.at[3, 7].set(G * dt); A = A.at[4, 6].set(-G * dt)

    B = jnp.zeros((12, 4))
    B = B.at[5, 0].set(dt / MASS)
    B = B.at[9:12, 1:4].set(jnp.diag(1/J_DIAG) * dt)
    return A, B

# =========================================================
# 3. Dense QP Projector 
# =========================================================
class KoopmanWallProjector:
    def __init__(self, horizon, n_cp, dt, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.dim_u = 4
        
        A_lin, B_lin = get_linear_model(dt)
        self.dim_z = A_lin.shape[0]
        self.Phi, self.Gamma = self._precompute_dense_matrices(A_lin, B_lin, horizon)
        
        base_mat = bspline_gen.basis_mat
        self.M_spline = jnp.kron(base_mat, jnp.eye(self.dim_u))
        self.K_mat = self.Gamma @ self.M_spline 
        
        self.solver = BatchedADMM(n_vars=n_cp * 4, rho=5.0, max_iter=30)
        
        self.u_min = jnp.array([0.0, -1.0, -1.0, -0.2])
        self.u_max = jnp.array([2.0*MASS*G, 1.0, 1.0, 0.2])

    def _precompute_dense_matrices(self, A, B, H):
        dim_z, dim_u = B.shape
        A_pows = [jnp.eye(dim_z)]
        for _ in range(H):
            A_pows.append(A @ A_pows[-1])
        Phi = jnp.vstack(A_pows[1:])
        col_blocks = []
        for i in range(H):
            col = []
            for j in range(H):
                if i >= j:
                    col.append(A_pows[i-j] @ B)
                else:
                    col.append(jnp.zeros((dim_z, dim_u)))
            col_blocks.append(jnp.hstack(col))
        Gamma = jnp.vstack(col_blocks) 
        return Phi, Gamma

    @partial(jax.jit, static_argnums=(0,))
    def get_trajectory_flat(self, coeffs_flat, z0):
        return self.Phi @ z0 + self.K_mat @ coeffs_flat

    def get_window_bounds(self, x_traj, window_center, window_size, wall_range, margin):
        """
        Margin을 고려하여 드론의 중심(Point Mass)이 갈 수 있는 유효 공간을 계산합니다.
        Effective Size = Actual Window Size - 2 * Margin
        """
        wall_start, wall_end = wall_range
        
        # 터널 구간
        in_tunnel = (x_traj >= wall_start) & (x_traj <= wall_end)
        
        y_center, z_center = window_center
        w_y, w_z = window_size
        
        # [핵심 변경] 터널 안에서의 허용 범위 = (창문 크기 / 2) - 마진
        # 마진만큼 벽에서 떨어져야 함
        safe_half_y = jnp.maximum(w_y/2.0 - margin, 0.01) # 최소폭 보장
        safe_half_z = jnp.maximum(w_z/2.0 - margin, 0.01)
        
        y_bound = jnp.where(in_tunnel, safe_half_y, 4.0) 
        
        # 천장과 바닥도 마진 적용
        z_bound_u = jnp.where(in_tunnel, z_center + safe_half_z, 6.0)
        z_bound_l = jnp.where(in_tunnel, z_center - safe_half_z, 0.0)
        
        return y_bound, z_bound_l, z_bound_u

    @partial(jax.jit, static_argnums=(0,))
    def project_single_sample(self, coeffs_noisy, z0, window_center, window_size, wall_range, margin):
        coeffs_flat = coeffs_noisy.reshape(-1)
        traj_flat = self.get_trajectory_flat(coeffs_flat, z0)
        traj = traj_flat.reshape(self.H, self.dim_z)
        pos_x = traj[:, 0]
        
        # Margin을 포함하여 Bound 계산
        y_half_bound, z_min, z_max = self.get_window_bounds(pos_x, window_center, window_size, wall_range, margin)
        
        K_tensor = self.K_mat.reshape(self.H, self.dim_z, -1)
        K_y = K_tensor[:, 1, :] 
        K_z = K_tensor[:, 2, :] 
        
        const_traj = (self.Phi @ z0).reshape(self.H, self.dim_z)
        const_y = const_traj[:, 1]
        const_z = const_traj[:, 2]

        # Constraints Setup
        A_y_upper = K_y
        b_y_upper = y_half_bound - const_y
        A_y_lower = -K_y
        b_y_lower = y_half_bound + const_y 

        A_z_upper = K_z
        b_z_upper = z_max - const_z
        A_z_lower = -K_z
        b_z_lower = -z_min + const_z
        
        # Input Constraints
        A_inp = jnp.eye(self.N_cp * 4)
        l_inp = jnp.tile(self.u_min, self.N_cp) - coeffs_flat
        u_inp = jnp.tile(self.u_max, self.N_cp) - coeffs_flat

        A_ineq = jnp.vstack([A_y_upper, A_y_lower, A_z_upper, A_z_lower])
        u_ineq = jnp.concatenate([b_y_upper, b_y_lower, b_z_upper, b_z_lower])
        l_ineq = jnp.full_like(u_ineq, -1e9)
        
        A = jnp.vstack([A_ineq, A_inp])
        l = jnp.concatenate([l_ineq, l_inp])
        u = jnp.concatenate([u_ineq, u_inp])
        
        P = jnp.eye(self.N_cp * 4)
        q = jnp.zeros(self.N_cp * 4)
        
        delta, _ = self.solver.solve(P, q, A, l, u)
        return (coeffs_flat + delta).reshape(self.N_cp, 4)

# =========================================================
# 4. Koopman MPPI (With Margin Cost)
# =========================================================
class KoopmanMPPI:
    def __init__(self, horizon, n_cp, dt, n_samples, temp):
        self.H = horizon
        self.N_cp = n_cp
        self.K = n_samples
        self.temp = temp
        self.bspline = BSplineBasis(n_cp, horizon)
        self.projector = KoopmanWallProjector(horizon, n_cp, dt, self.bspline)
        self.dt = dt

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, mean_coeffs, z_curr, target, win_center, win_size, wall_range, margin):
        std = jnp.array([1.5, 0.5, 0.5, 0.2]) 
        noise = jax.random.normal(key, (self.K, self.N_cp, 4)) * std
        samples = mean_coeffs + noise
        
        # Projector에도 Margin 전달
        safe_samples = jax.vmap(self.projector.project_single_sample, in_axes=(0, None, None, None, None, None))(
            samples, z_curr, win_center, win_size, wall_range, margin
        )
        
        costs = jax.vmap(self.compute_cost, in_axes=(0, None, None, None, None, None, None))(
            safe_samples, z_curr, target, win_center, win_size, wall_range, margin
        )
        
        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.temp)
        new_mean = jnp.sum(weights[:, None, None] * safe_samples, axis=0)
        
        return 0.8 * new_mean + 0.2 * mean_coeffs, safe_samples, weights

    def compute_cost(self, coeffs, z0, target, win_center, win_size, wall_range, margin):
        u_seq = self.bspline.get_sequence(coeffs)
        
        def scan_fn(z, u):
            z_next = se3_step(z, u, self.dt)
            return z_next, z_next
        _, traj = jax.lax.scan(scan_fn, z0, u_seq)
        
        pos = traj[:, :3]
        vel = traj[:, 3:6]
        
        dist_err = jnp.sum((pos - target)**2)
        final_err = jnp.sum((pos[-1] - target)**2) * 20.0
        
        # --- Tunnel Violation Cost (With Margin) ---
        x, y, z = pos[:, 0], pos[:, 1], pos[:, 2]
        yc, zc = win_center
        wy, wz = win_size
        wall_start, wall_end = wall_range
        
        in_tunnel = (x >= wall_start) & (x <= wall_end)
        
        # 물리적 벽까지의 거리가 margin보다 작아지면 비용 발생
        # Violation = |pos - center| - (Half_Width - Margin)
        # 즉, 안전 영역(중심에서 Half_Width - Margin)을 벗어나면 페널티
        
        safe_hy = wy/2.0 - margin
        safe_hz = wz/2.0 - margin
        
        y_viol = jnp.maximum(jnp.abs(y) - safe_hy, 0.0)
        z_viol = jnp.maximum(jnp.abs(z - zc) - safe_hz, 0.0)
        
        tunnel_cost = jnp.sum(in_tunnel * (jnp.exp(30.0 * y_viol) + jnp.exp(30.0 * z_viol) - 2.0))
        
        vel_cost = jnp.sum(vel**2) * 0.01
        
        return dist_err + final_err + 20.0 * tunnel_cost + vel_cost

# =========================================================
# 5. Main Simulation & Visualization (Sphere Draw)
# =========================================================

def draw_tunnel(ax, wall_range, win_center, win_size):
    x_start, x_end = wall_range
    yc, zc = win_center
    wy, wz = win_size
    hy, hz = wy/2, wz/2
    
    x = np.linspace(x_start, x_end, 10)
    
    # Visual은 "물리적 벽"을 그립니다 (Margin 적용 전 실제 벽 위치)
    
    # 1. Floor & Ceiling
    y = np.linspace(yc - hy, yc + hy, 2)
    X, Y = np.meshgrid(x, y)
    Z_floor = np.full_like(X, zc - hz)
    Z_ceil = np.full_like(X, zc + hz)
    ax.plot_surface(X, Y, Z_floor, color='gray', alpha=0.2)
    ax.plot_surface(X, Y, Z_ceil, color='gray', alpha=0.2)
    
    # 2. Walls
    z = np.linspace(zc - hz, zc + hz, 2)
    X, Z = np.meshgrid(x, z)
    Y_left = np.full_like(X, yc - hy)
    Y_right = np.full_like(X, yc + hy)
    ax.plot_surface(X, Y_left, Z, color='k', alpha=0.1)
    ax.plot_surface(X, Y_right, Z, color='k', alpha=0.1)
    
    # Edges
    for x_pos in [x_start, x_end]:
        ys = [yc-hy, yc+hy, yc+hy, yc-hy, yc-hy]
        zs = [zc-hz, zc-hz, zc+hz, zc+hz, zc-hz]
        ax.plot([x_pos]*5, ys, zs, 'k-', linewidth=1.5)

def draw_drone_sphere(ax, pos, radius):
    """드론의 크기(Margin)를 구 형태로 시각화"""
    u = np.linspace(0, 2 * np.pi, 10)
    v = np.linspace(0, np.pi, 10)
    x = radius * np.outer(np.cos(u), np.sin(v)) + pos[0]
    y = radius * np.outer(np.sin(u), np.sin(v)) + pos[1]
    z = radius * np.outer(np.ones(np.size(u)), np.cos(v)) + pos[2]
    ax.plot_wireframe(x, y, z, color='b', alpha=0.3)

def plot_profiles(history_dict, dt):
    """
    RPY, Thrust, Torques 프로파일링 시각화 함수
    """
    rpy = np.array(history_dict['rpy'])
    u = np.array(history_dict['u'])
    
    # 시간 축 생성
    time_steps = np.arange(len(rpy)) * dt
    
    fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    
    # 1. Attitude (RPY)
    axes[0].plot(time_steps, rpy[:, 0], label='Roll (phi)', color='r')
    axes[0].plot(time_steps, rpy[:, 1], label='Pitch (theta)', color='g')
    axes[0].plot(time_steps, rpy[:, 2], label='Yaw (psi)', color='b')
    axes[0].set_ylabel('Angle [rad]')
    axes[0].set_title('Drone Attitude (Roll, Pitch, Yaw)')
    axes[0].legend(loc='upper right')
    axes[0].grid(True)
    
    # 2. Control Input: Thrust
    axes[1].plot(time_steps, u[:, 0], label='Thrust', color='k')
    # Saturation Line (Max Thrust)
    axes[1].axhline(y=20.0, color='r', linestyle='--', alpha=0.5, label='Max Thrust')
    axes[1].set_ylabel('Thrust [N]')
    axes[1].set_title('Control Input: Thrust')
    axes[1].legend(loc='upper right')
    axes[1].grid(True)
    
    # 3. Control Input: Torques
    axes[2].plot(time_steps, u[:, 1], label='Torque X', color='r', linestyle='--')
    axes[2].plot(time_steps, u[:, 2], label='Torque Y', color='g', linestyle='--')
    axes[2].plot(time_steps, u[:, 3], label='Torque Z', color='b', linestyle='--')
    # Saturation Line (Max Torque)
    axes[2].axhline(y=1.0, color='gray', linestyle=':', alpha=0.5)
    axes[2].axhline(y=-1.0, color='gray', linestyle=':', alpha=0.5)
    axes[2].set_ylabel('Torque [Nm]')
    axes[2].set_xlabel('Time [s]')
    axes[2].set_title('Control Input: Torques')
    axes[2].legend(loc='upper right')
    axes[2].grid(True)
    
    plt.tight_layout()
    plt.show()

def run_simulation(save_gif=True, gif_filename="drone_corridor.gif"):
    DT = 0.01
    H = 40
    N_CP = 10
    K = 128
    
    mppi = KoopmanMPPI(H, N_CP, DT, K, 1.0)
    
    z_curr = jnp.zeros(12)
    z_curr = z_curr.at[2].set(1.0) 
    
    # --- 시나리오 설정 ---
    wall_range = jnp.array([2.0, 6.0]) # 긴 터널
    target = jnp.array([8.0, 0.0, 2.0])
    win_center = jnp.array([0.0, 2.0])
    win_size = jnp.array([0.8, 0.8])
    
    MARGIN = 0.2
    
    mean_coeffs = jnp.zeros((N_CP, 4))
    mean_coeffs = mean_coeffs.at[:, 0].set(MASS * G)
    
    key = jax.random.PRNGKey(42)
    
    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    traj_hist = [z_curr[:3]]
    
    frames = []
    profile_data = {'rpy': [], 'u': []}

    print(f"Simulation Running...")
    
    for t in range(400):
        key, subkey = jax.random.split(key)
        
        t0 = time.time()
        # Margin 전달
        mean_coeffs, samples, weights = mppi.step(
            subkey, mean_coeffs, z_curr, target, win_center, win_size, wall_range, MARGIN
        )
        jax.block_until_ready(mean_coeffs)
        calc_time = time.time() - t0
        
        u_applied = mppi.bspline.get_sequence(mean_coeffs)[0]
        z_curr = se3_step(z_curr, u_applied, DT)
        traj_hist.append(z_curr[:3])
        
        profile_data['rpy'].append(z_curr[6:9])
        profile_data['u'].append(u_applied)

        dist = jnp.linalg.norm(z_curr[:3] - target)
        
        if t % 1 == 0:
            ax.cla()
            
            draw_tunnel(ax, wall_range, win_center, win_size)
            
            # [추가] 드론을 구(Sphere)로 그리기
            draw_drone_sphere(ax, z_curr[:3], MARGIN - 0.15)
            
            top_idx = np.argsort(np.array(weights))[-20:]
            for idx in top_idx:
                c_flat = samples[idx].reshape(-1)
                traj_flat = mppi.projector.get_trajectory_flat(c_flat, z_curr)
                traj = traj_flat.reshape(H, 12)
                ax.plot(traj[:,0], traj[:,1], traj[:,2], 'g-', alpha=0.1)
                
            hist = np.array(traj_hist)
            ax.plot(hist[:,0], hist[:,1], hist[:,2], 'b-', linewidth=2)
            ax.scatter(target[0], target[1], target[2], c='r', marker='*', s=200)
            
            ax.set_xlim(-1, 9)
            ax.set_ylim(-3, 3)
            ax.set_zlim(0, 5)
            
            if save_gif:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=180)
                buf.seek(0)
                frames.append(imageio.imread(buf))
                buf.close()
            else:
                plt.pause(0.01)
            
        if dist < 0.2:
            print("Target Reached!")
            break
            
    if save_gif and len(frames) > 0:
        imageio.mimsave(gif_filename, frames, fps=8, loop=0)
        print("Done!")
    else: 
        plt.show()

    plot_profiles(profile_data, DT)

if __name__ == "__main__":
    run_simulation(save_gif=True, gif_filename="drone_corridor.gif")