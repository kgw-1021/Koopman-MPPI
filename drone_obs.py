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
            # Simple RBF approximation for B-Spline
            val = jnp.exp(-0.5 * ((t - center) / 0.15)**2)
            basis.append(val)
        basis = jnp.stack(basis, axis=1)
        basis = basis / (jnp.sum(basis, axis=1, keepdims=True) + 1e-6)
        return basis

    @partial(jax.jit, static_argnums=(0,))
    def get_sequence(self, coeffs):
        return jnp.dot(self.basis_mat, coeffs)

# =========================================================
# 2. Dynamics (Linear & Nonlinear)
# =========================================================

@jax.jit
def se3_step(state, u, dt):
    """ Nonlinear Dynamics with Damping """
    pos, vel, rpy, omega = state[0:3], state[3:6], state[6:9], state[9:12]
    thrust, torques = u[0], u[1:]
    
    # --- [Fix 1] Clamp Constraints (물리적으로 불가능한 값 방지) ---
    thrust = jnp.clip(thrust, 0.0, 20.0) # 0 ~ 2*mg
    torques = jnp.clip(torques, -1.0, 1.0)
    
    # Rotation Matrix
    phi, theta, psi = rpy
    cphi, sphi = jnp.cos(phi), jnp.sin(phi)
    cth, sth = jnp.cos(theta), jnp.sin(theta)
    cpsi, spsi = jnp.cos(psi), jnp.sin(psi)
    
    R_vec = jnp.array([
        cpsi*cth, cpsi*sth*sphi - spsi*cphi, cpsi*sth*cphi + spsi*sphi,
        spsi*cth, spsi*sth*sphi + cpsi*cphi, spsi*sth*cphi - cpsi*sphi,
        -sth,     cth*sphi,                  cth*cphi
    ]).reshape(3,3)

    # --- [Fix 2] Add Air Damping (공기 저항) ---
    # 이게 없으면 에너지가 무한정 발산합니다.
    drag_lin = -0.1 * vel  # 선형 저항
    drag_ang = -0.1 * omega # 각속도 저항
    
    # Dynamics
    acc = (R_vec @ jnp.array([0., 0., thrust])) / MASS - jnp.array([0., 0., G]) + drag_lin
    
    # Gyroscopic effect + Torque + Damping
    omega_dot = J_INV * (torques - jnp.cross(omega, J_DIAG * omega) + drag_ang)
    
    # Integration
    next_pos = pos + vel * dt
    next_vel = vel + acc * dt
    next_rpy = rpy + omega * dt # Small angle assumption for kinematics
    next_omega = omega + omega_dot * dt
    
    return jnp.concatenate([next_pos, next_vel, next_rpy, next_omega])

def get_linear_model(dt):
    """ Linearized Koopman Model (Hover) """
    A = jnp.eye(12)
    A = A.at[0:3, 3:6].set(jnp.eye(3) * dt) # pos -> vel
    A = A.at[6:9, 9:12].set(jnp.eye(3) * dt) # angle -> omega
    A = A.at[3, 7].set(G * dt); A = A.at[4, 6].set(-G * dt) # Small angle gravity

    B = jnp.zeros((12, 4))
    B = B.at[5, 0].set(dt / MASS) # Thrust
    B = B.at[9:12, 1:4].set(jnp.diag(1/J_DIAG) * dt) # Torques
    return A, B

# =========================================================
# 3. Dense QP Projector (With FIX applied)
# =========================================================
class KoopmanDenseProjector:
    def __init__(self, horizon, n_cp, dt, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.dim_u = 4
        
        # 1. Linear System
        A_lin, B_lin = get_linear_model(dt)
        self.dim_z = A_lin.shape[0]
        
        # 2. Precompute Phi, Gamma (Dense Matrices)
        # Z = Phi * z0 + Gamma * U
        self.Phi, self.Gamma = self._precompute_dense_matrices(A_lin, B_lin, horizon)
        
        # 3. Combine with B-Spline Matrix
        base_mat = bspline_gen.basis_mat # (H, N_cp)
        self.M_spline = jnp.kron(base_mat, jnp.eye(self.dim_u)) # (H*4, N_cp*4)
        
        self.K_mat = self.Gamma @ self.M_spline # (H*12, N_cp*4) Constants!
        
        # Solver & Constraints
        # user provided ADMM is used here
        self.solver = BatchedADMM(n_vars=n_cp * 4, rho=2.0, max_iter=20)
        
        self.u_min = jnp.array([0.0, -1.0, -1.0, -0.2])
        self.u_max = jnp.array([2.0*MASS*G, 1.0, 1.0, 0.2])

    def _precompute_dense_matrices(self, A, B, H):
        dim_z, dim_u = B.shape
        # Compute A^k
        A_pows = [jnp.eye(dim_z)]
        for _ in range(H):
            A_pows.append(A @ A_pows[-1])
        
        # Phi: Stack of A^k [A; A^2; ... A^H]
        Phi = jnp.vstack(A_pows[1:])
        
        # Gamma: Block convolution matrix
        # (Constructing strictly lower triangular block matrix)
        # Using a simple loop for initialization (runs once)
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
        # O(1) Matrix Mul
        return self.Phi @ z0 + self.K_mat @ coeffs_flat

    @partial(jax.jit, static_argnums=(0,))
    def project_single_sample(self, coeffs_noisy, z0, obs_pos, obs_r):
        """ Dense Projection with ADMM Unpacking Fix """
        coeffs_flat = coeffs_noisy.reshape(-1)
        
        # 1. Rollout
        traj_flat = self.get_trajectory_flat(coeffs_flat, z0)
        traj = traj_flat.reshape(self.H, self.dim_z)
        pos_traj = traj[:, :3] # (H, 3)
        
        # 2. Obstacle Constraint Linearization
        diff = pos_traj - obs_pos
        dist_sq = jnp.sum(diff**2, axis=1)
        dist = jnp.sqrt(dist_sq + 1e-6)
        normals = diff / dist[:, None] # (H, 3)
        
        # Jacobian J = Slices of K_mat
        K_pos = self.K_mat.reshape(self.H, self.dim_z, -1)[:, :3, :] # (H, 3, N_vars)
        
        # A_obs = -n^T * J
        A_obs = -jnp.einsum('td,tdv->tv', normals, K_pos) # (H, N_vars)
        b_obs = -(obs_r + 0.3 - dist) # Margin 0.3
        l_obs = jnp.full_like(b_obs, -1e9)
        
        # 3. Input Constraints
        A_inp = jnp.eye(self.N_cp * 4)
        l_inp = jnp.tile(self.u_min, self.N_cp) - coeffs_flat
        u_inp = jnp.tile(self.u_max, self.N_cp) - coeffs_flat
        
        # 4. Stack
        A = jnp.vstack([A_obs, A_inp])
        l = jnp.concatenate([l_obs, l_inp])
        u = jnp.concatenate([b_obs, u_inp])
        
        P = jnp.eye(self.N_cp * 4)
        q = jnp.zeros(self.N_cp * 4)
        
        # [수정됨] ADMM Solve 호출 및 Unpacking
        # ADMM returns: x_final, (x_final, z_final, y_final)
        # 우리는 x_final(delta)만 필요합니다.
        delta, _ = self.solver.solve(P, q, A, l, u)
        
        return (coeffs_flat + delta).reshape(self.N_cp, 4)

# =========================================================
# 4. Koopman MPPI
# =========================================================
class KoopmanMPPI:
    def __init__(self, horizon, n_cp, dt, n_samples, temp):
        self.H = horizon
        self.N_cp = n_cp
        self.K = n_samples
        self.temp = temp
        self.bspline = BSplineBasis(n_cp, horizon)
        self.projector = KoopmanDenseProjector(horizon, n_cp, dt, self.bspline)
        self.dt = dt

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, mean_coeffs, z_curr, target, obs_pos, obs_r):
        # 1. Sampling
        std = jnp.array([1.0, 0.3, 0.3, 0.1])
        noise = jax.random.normal(key, (self.K, self.N_cp, 4)) * std
        samples = mean_coeffs + noise
        
        # 2. Dense Projection (VMAP)
        # project_single_sample now correctly handles ADMM return tuple
        safe_samples = jax.vmap(self.projector.project_single_sample, in_axes=(0, None, None, None))(
            samples, z_curr, obs_pos, obs_r
        )
        
        # 3. Cost Calculation
        costs = jax.vmap(self.compute_cost, in_axes=(0, None, None, None, None))(
            safe_samples, z_curr, target, obs_pos, obs_r
        )
        
        # 4. Update
        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.temp)
        new_mean = jnp.sum(weights[:, None, None] * safe_samples, axis=0)
        
        return 0.8 * new_mean + 0.2 * mean_coeffs, safe_samples, weights

    def compute_cost(self, coeffs, z0, target, obs_pos, obs_r):
        u_seq = self.bspline.get_sequence(coeffs)
        
        # Nonlinear Scan for accurate cost
        def scan_fn(z, u):
            z_next = se3_step(z, u, self.dt)
            return z_next, z_next
        _, traj = jax.lax.scan(scan_fn, z0, u_seq)
        
        pos = traj[:, :3]
        vel = traj[:, 3:6]  
        dist_err = jnp.sum((pos - target)**2)
        vel_running = jnp.sum(vel**2) * 0.05
        vel_terminal = jnp.sum(vel[-1]**2) * 30.0
        final_err = jnp.sum((pos[-1] - target)**2) * 50.0
        # Obstacle Cost (Soft penalty for MPPI selection)
        obs_dist = jnp.linalg.norm(pos - obs_pos, axis=1)
        obs_pen = jnp.sum(jnp.exp(-2.0*(obs_dist - obs_r))) * 50.0
        
        return dist_err + final_err + obs_pen + vel_running + vel_running + vel_terminal 

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

# =========================================================
# 5. Main Simulation
# =========================================================
def run_simulation(save_gif=True, gif_filename="drone_obstacle.gif"):
    # Params
    DT = 0.01      
    H = 40         
    N_CP = 10
    K = 128
    
    mppi = KoopmanMPPI(H, N_CP, DT, K, 0.8)
    
    # Init State: x,y,z=0, vx,vy,vz=0, rpy=0, omega=0
    z_curr = jnp.zeros(12)
    
    # Scenario
    target = jnp.array([3.0, 3.0, 3.0])
    obs_pos = jnp.array([1.5, 1.5, 1.5])
    obs_r = 1.0
    
    mean_coeffs = jnp.zeros((N_CP, 4))
    mean_coeffs = mean_coeffs.at[:, 0].set(MASS * G)
    
    key = jax.random.PRNGKey(0)
    
    # Visualization
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    traj_hist = [z_curr[:3]]
    
    frames = []
    profile_data = {'rpy': [], 'u': []}


    print("Simulation Running...")
    
    for t in range(500):
        key, subkey = jax.random.split(key)
        
        t0 = time.time()
        # MPPI Step
        mean_coeffs, samples, weights = mppi.step(subkey, mean_coeffs, z_curr, target, obs_pos, obs_r)
        
        # Block valid for timing accurate measurements
        jax.block_until_ready(mean_coeffs)
        dt_step = time.time() - t0
        
        # Apply Control
        u_applied = mppi.bspline.get_sequence(mean_coeffs)[0]
        z_curr = se3_step(z_curr, u_applied, DT)
        traj_hist.append(z_curr[:3])
        
        profile_data['rpy'].append(z_curr[6:9])
        profile_data['u'].append(u_applied)

        dist = jnp.linalg.norm(z_curr[:3] - target)
        
        if t % 1 == 0:
            ax.cla()
            # Draw Obstacle
            u, v = np.mgrid[0:2*np.pi:20j, 0:np.pi:10j]
            x_o = obs_r*np.cos(u)*np.sin(v) + obs_pos[0]
            y_o = obs_r*np.sin(u)*np.sin(v) + obs_pos[1]
            z_o = obs_r*np.cos(v) + obs_pos[2]
            ax.plot_wireframe(x_o, y_o, z_o, color='r', alpha=0.1)
            
            # Draw Candidates (Predicted via Linear Model for speed viz)
            # Just visualizing top 3
            top_idx = np.argsort(np.array(weights))[-30:]
            for idx in top_idx:
                c_flat = samples[idx].reshape(-1)
                traj_flat = mppi.projector.get_trajectory_flat(c_flat, z_curr)
                traj = traj_flat.reshape(H, 12)
                ax.plot(traj[:,0], traj[:,1], traj[:,2], 'g-', alpha=0.3)
                
            # History
            hist = np.array(traj_hist)
            ax.plot(hist[:,0], hist[:,1], hist[:,2], 'k-', linewidth=2, label='Path')
            ax.scatter(target[0], target[1], target[2], marker='*', s=200, color='gold')
            
            ax.set_xlim(-1, 4); ax.set_ylim(-1, 4); ax.set_zlim(0, 4)

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
    run_simulation(save_gif=False, gif_filename="drone_obstacle.gif")