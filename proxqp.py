import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from Bspline import BSplineBasis
import time

# [NEW] JaxProxQP Imports
from jaxproxqp.jaxproxqp import JaxProxQP
from jaxproxqp.qp_problems import QPModel
from jaxproxqp.settings import Settings

# =========================================================
# 1. System Dynamics (Dynamics with Acceleration)
# =========================================================

@jax.jit
def lift_state(state_std):
    """
    state_std: [x, y, theta, v, w]
    Returns:   [x, y, cos, sin, v, w]
    """
    x, y, theta, v, w = state_std
    return jnp.array([x, y, jnp.cos(theta), jnp.sin(theta), v, w])

@jax.jit
def step_(state, control, dt):
    """
    state:   [x, y, c, s, v, w]
    control: [accel_v, accel_w]
    """
    x, y, c, s, v, w = state
    av, aw = control

    # 1. 속도 업데이트
    next_v = v + av * dt
    next_w = w + aw * dt

    # 2. 위치 업데이트 (Kinematics)
    rot_c = jnp.cos(next_w * dt)
    rot_s = jnp.sin(next_w * dt)

    next_c = c * rot_c - s * rot_s
    next_s = s * rot_c + c * rot_s
    
    next_x = x + next_v * next_c * dt
    next_y = y + next_v * next_s * dt

    return jnp.array([next_x, next_y, next_c, next_s, next_v, next_w])

# =========================================================
# 2. ProxQP Projector (Using jaxproxqp)
# =========================================================

class ProxQPProjector:
    def __init__(self, horizon, n_cp, dt, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.bspline = bspline_gen
        self.dim_var = n_cp * 2

        # [Solver Settings]
        # 정확도와 속도 균형을 위한 설정
        self.settings = Settings()
        self.settings.max_iter = 50
        self.settings.eps_abs = 1e-2    

        # [Input Constraints] Box Constraints for jaxproxqp
        self.u_min = jnp.array([-1.0, -2.0]) 
        self.u_max = jnp.array([ 1.0,  2.0])

    @partial(jax.jit, static_argnums=(0,))
    def rollout_fn(self, coeffs, z0):
        u_seq = self.bspline.get_sequence(coeffs)
        
        def step_fn(carry, u):
            z_next = step_(carry, u, self.dt)
            return z_next, z_next # Return full state
            
        _, traj = jax.lax.scan(step_fn, z0, u_seq)
        return traj

    @partial(jax.jit, static_argnums=(0,))
    def jac_fn(self, coeffs, z0):
        return jax.jacfwd(self.rollout_fn, argnums=0)(coeffs, z0)

    @partial(jax.jit, static_argnums=(0,))
    def project_single_sample(self, coeffs_noisy, z0, obs_pos, obs_r, target_pos):
        """
        Samples projection using JaxProxQP with Warm Start.
        
        minimize 0.5 * ||x - coeffs_noisy||^2
        s.t.     l_box <= x <= u_box
                 A x = b          (Terminal State Equality)
                 l <= C x <= u    (Obstacle Avoidance Inequality)
        """
        # 1. Trajectory & Jacobian
        p_traj = self.rollout_fn(coeffs_noisy, z0)      # (H, 6)
        J_tensor = self.jac_fn(coeffs_noisy, z0)        # (H, 6, N_CP, 2)

        # ---------------------------------------------------
        # [Objective] minimize ||x - coeffs_noisy||^2
        # => 0.5 * x' I x - coeffs_noisy' x
        # ---------------------------------------------------
        H_mat = jnp.eye(self.dim_var)
        g_vec = -coeffs_noisy.flatten()

        # ---------------------------------------------------
        # [Constraint 1] Box Constraints (l_box, u_box)
        # ---------------------------------------------------
        l_box = jnp.tile(self.u_min, self.N_cp)
        u_box = jnp.tile(self.u_max, self.N_cp)

        # ---------------------------------------------------
        # [Constraint 2] Equality Constraints (A x = b)
        # Terminal State (Zero Velocity) at step H
        # ---------------------------------------------------
        v_final = p_traj[-1, 4]
        w_final = p_traj[-1, 5]
        
        J_v_end = J_tensor[-1, 4].reshape(1, -1)
        J_w_end = J_tensor[-1, 5].reshape(1, -1)
        
        # Linearized: J * delta = -val  =>  A * x = b
        # 주의: x는 delta가 아니라 전체 coeffs임에 주의해야 하나, 
        # 여기서는 편의상 x를 전체 coeffs로 두고 QP를 풀 수 있도록
        # A * (coeffs_noisy + delta) = 0 가 되도록 설정하는 것이 아니라
        # linearization around coeffs_noisy: 
        # val + J * (x - coeffs_noisy) = 0  =>  J * x = J * coeffs_noisy - val
        
        # A matrix
        A_eq = jnp.vstack([J_v_end, J_w_end]) # (2, dim_var)
        
        # b vector
        # Current val: [v_final, w_final]
        # Target: 0
        # Linear approx: val + J * delta = 0  => J * delta = -val
        # x = x_nominal + delta => delta = x - x_nominal
        # J * (x - x_nominal) = -val
        # J * x = J * x_nominal - val
        b_eq = A_eq @ coeffs_noisy.flatten() - jnp.array([v_final, w_final])

        # ---------------------------------------------------
        # [Constraint 3] Inequality Constraints (l <= C x <= u)
        # Obstacle Avoidance
        # ---------------------------------------------------
        pos_traj = p_traj[:, :2]
        diff = pos_traj - obs_pos 
        dist_sq = jnp.sum(diff**2, axis=1)              
        dist_vals = jnp.sqrt(dist_sq) + 1e-6            
        normals = diff / dist_vals[:, None]             # (H, 2)

        J_pos = J_tensor[:, :2, :, :] 
        J_flat = J_pos.reshape(self.H, 2, self.dim_var)
        
        # Linearization: dist >= req_dist
        # dist_val + n^T * J * (x - x_nom) >= req_dist
        # n^T * J * x >= req_dist - dist_val + n^T * J * x_nom
        
        # C matrix (H rows)
        C_ineq = jnp.einsum('td,tdv->tv', normals, J_flat)
        
        safe_margin = 0.3
        req_dist = obs_r + safe_margin
        
        # Lower bound for C*x
        l_ineq = req_dist - dist_vals + (C_ineq @ coeffs_noisy.flatten())
        # Upper bound (infinity)
        u_ineq = jnp.full_like(l_ineq, 1e5) 

        # ---------------------------------------------------
        # [Create QP Model]
        # ---------------------------------------------------
        # Note: QPModel.create supports A, b for equalities if implemented in the repo version.
        # If not supported in your specific version, map A/b to C/l/u with tight bounds.
        # Assuming standard ProxQP formulation structure:
        qp = QPModel.create(
            H=H_mat, g=g_vec,
            A=A_eq, b=b_eq,            # Equality
            C=C_ineq, l=l_ineq, u=u_ineq, # Inequality
            l_box=l_box, u_box=u_box
        )

        # ---------------------------------------------------
        # [Solve with Warm Start]
        # ---------------------------------------------------
        solver = JaxProxQP(qp, self.settings)
        
        # Warm Start Strategy:
        # init_x: 우리는 x_nominal(coeffs_noisy) 주변에서 해를 찾고 있으므로, 
        #         coeffs_noisy 자체가 훌륭한 Primal 초기값입니다.
        # init_y: 등식 제약(속도)에 대한 Dual 변수 (0으로 초기화)
        # init_z: 부등식 제약(장애물)에 대한 Dual 변수 (0으로 초기화)
        
        n_eq = A_eq.shape[0]
        n_ineq = C_ineq.shape[0]
        
        # solve() 메서드가 init_x, init_y, init_z를 지원한다고 가정 (ProxQP 표준)
        sol = solver.solve()
        
        safe_coeffs = sol.x.reshape(self.N_cp, 2)
        return safe_coeffs, sol.x # Return raw x if needed later

# =========================================================
# 3. ProxQP MPPI (MPPI Logic Wrapper)
# =========================================================

class ProxMPPI:
    def __init__(self, horizon, n_cp, dt, n_samples, temperature, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.K = n_samples
        self.lambda_ = temperature
        self.projector = ProxQPProjector(horizon, n_cp, dt, bspline_gen)

    @partial(jax.jit, static_argnums=(0,))
    def compute_cost(self, coeffs, z0, target_pos, obs_pos, obs_r):
        # traj: [x, y, c, s, v, w]
        traj = self.projector.rollout_fn(coeffs, z0)
        pos_traj = traj[:, :2]
        vel_traj = traj[:, 4:] # v, w
        
        # 1. 위치 오차
        term_err = jnp.sum((pos_traj[-1] - target_pos)**2)
        stage_err = jnp.sum(jnp.sum((pos_traj - target_pos)**2, axis=1))
        
        # 2. 장애물 비용
        dist = jnp.sqrt(jnp.sum((pos_traj - obs_pos)**2, axis=1))
        obs_cost = jnp.sum(jnp.exp(-5.0 * (dist - obs_r - 0.2)))
        
        # 3. 부드러움 (Jerk)
        diffs = jnp.diff(coeffs, axis=0) 
        smoothness_cost = jnp.sum(diffs**2)
        
        # 4. 에너지 및 종단 정지 비용
        energy_cost = jnp.sum(coeffs**2) 
        terminal_vel_cost = jnp.sum(vel_traj[-1]**2)

        total_cost = (
            30.0 * term_err + 
            1.0 * stage_err + 
            10.0 * obs_cost + 
            5.0 * smoothness_cost +
            0.1 * energy_cost +
            30.0 * terminal_vel_cost 
        )
        return total_cost

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, mean_coeffs, z0, target_pos, obs_pos, obs_r):
        # 1. Adaptive Noise
        dist_to_goal = jnp.linalg.norm(z0[:2] - target_pos)
        sigma = jnp.where(dist_to_goal < 1.0, 0.2, 0.6)
        
        noise = jax.random.normal(key, (self.K, self.N_cp, 2)) * sigma
        raw_samples = mean_coeffs + noise
        
        # 2. Parallel Projection (using JaxProxQP)
        # vmap over K samples
        project_fn = jax.vmap(self.projector.project_single_sample, in_axes=(0, None, None, None, None))
        safe_samples, _ = project_fn(raw_samples, z0, obs_pos, obs_r, target_pos)
        
        # 3. Cost Evaluation
        cost_fn = jax.vmap(self.compute_cost, in_axes=(0, None, None, None, None))
        costs = cost_fn(safe_samples, z0, target_pos, obs_pos, obs_r)
        
        # 4. MPPI Update
        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.lambda_)
        
        weights_expanded = weights[:, None, None]
        new_mean = jnp.sum(weights_expanded * safe_samples, axis=0)
        
        alpha = 0.9 
        final_mean = alpha * new_mean + (1 - alpha) * mean_coeffs
        
        return final_mean, safe_samples, weights

# =========================================================
# Main Execution Loop
# =========================================================
def run():
    DT = 0.1
    HORIZON = 30
    N_CP = 10
    N_SAMPLES = 100  
    TEMP = 0.5      
    
    bspline_gen = BSplineBasis(N_CP, HORIZON)
    mppi = ProxMPPI(HORIZON, N_CP, DT, N_SAMPLES, TEMP, bspline_gen)
    
    # [초기 상태] x, y, theta, v, w
    start_pose = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    z_curr = lift_state(start_pose) 
    
    target_pos = jnp.array([5.0, 0.0])
    obs_pos = jnp.array([2.5, 0.0])
    obs_r = 0.8
    
    # 초기 제어 입력
    mean_coeffs = jnp.zeros((N_CP, 2))
    
    key = jax.random.PRNGKey(0)
    traj_hist = [z_curr[:2]]
    
    log_solver_time = [] # ms

    print("Simulation Running with JaxProxQP...")
    
    plt.figure(figsize=(10, 6))
    
    for t in range(300):
        key, subkey = jax.random.split(key)

        jax.block_until_ready(mean_coeffs)
        t0 = time.time()

        # 1. MPPI Step (ProxQP projection inside)
        mean_coeffs, safe_samples, weights = mppi.step(
            subkey, mean_coeffs, z_curr, target_pos, obs_pos, obs_r
        )
        
        jax.block_until_ready(mean_coeffs)
        t_end = time.time()

        solver_ms = (t_end - t0) * 1000.0
        log_solver_time.append(solver_ms)

        # 2. Apply Control
        u_seq = bspline_gen.get_sequence(mean_coeffs)
        u_curr = u_seq[0]
        
        dist_to_goal = jnp.linalg.norm(z_curr[:2] - target_pos)

        # 3. Simulate Dynamics
        z_curr = step_(z_curr, u_curr, DT)
        traj_hist.append(z_curr[:2])

        # 4. Visualization
        if t % 1 == 0:
            plt.clf()
            # Obstacle
            plt.gca().add_patch(plt.Circle(obs_pos, obs_r, color='r', alpha=0.5))
            plt.gca().add_patch(plt.Circle(obs_pos, obs_r + 0.3, color='r', fill=False, linestyle=':', label='Margin'))
            
            # Target
            plt.plot(target_pos[0], target_pos[1], 'bx', markersize=10, label='Target')
            
            # Candidates
            rollout_viz = jax.jit(jax.vmap(mppi.projector.rollout_fn, in_axes=(0, None)))
            top_idx = jnp.argsort(weights)[-10:] 
            top_samples = safe_samples[top_idx]
            top_trajs = rollout_viz(top_samples, z_curr) # (K, H, 6)
            
            for k in range(len(top_samples)):
                alp = 0.3 + 0.7 * (k / len(top_samples))
                plt.plot(top_trajs[k, :, 0], top_trajs[k, :, 1], 'g-', alpha=alp)
            
            # Mean Trajectory
            mean_traj = mppi.projector.rollout_fn(mean_coeffs, z_curr)
            plt.plot(mean_traj[:, 0], mean_traj[:, 1], 'b-', linewidth=3, label='ProxQP Path')
            
            # History
            hist = np.array(traj_hist)
            plt.plot(hist[:, 0], hist[:, 1], 'k--', label='Driven Path')
            plt.plot(z_curr[0], z_curr[1], 'ko')

            # Info
            vel_v = z_curr[4]
            vel_w = z_curr[5]
            plt.title(f"Step {t} | Dist: {dist_to_goal:.2f}m | Vel: {vel_v:.2f} m/s")
            plt.gca().set_aspect('equal', adjustable='box')
            plt.xlim(-5, 10)
            plt.ylim(-4, 4)
            plt.grid(True)
            plt.legend(loc='upper right')
            plt.pause(0.01)

        if dist_to_goal < 0.1:
            print("Goal Reached!")
            break

    plt.show()

    plt.figure()
    plt.plot(log_solver_time[3:], label='Solver Time (ms)')
    plt.title(f"JaxProxQP solver time per step")
    plt.xlabel("Simulation Step")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.show()

    # --- Final Stats Print ---
    print("\n" + "="*30)
    print(" [Simulation Result Summary (JaxProxQP)]")
    print("="*30)
    print(f"Avg Solver Time: {np.mean(log_solver_time[3:]):.2f} ms")
    print(f"Max Solver Time: {np.max(log_solver_time[3:]):.2f} ms")
    print(f"Min Solver Time: {np.min(log_solver_time[3:]):.2f} ms")
    print(f"Median Solver Time: {np.median(log_solver_time[3:]):.2f} ms")
    print("="*30)

if __name__ == "__main__":
    run()