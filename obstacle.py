import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from Bspline import BSplineBasis
from optimize import BatchedADMM 
import time
# =========================================================
# 1. System Dynamics (Dynamics with Acceleration)
# =========================================================

@jax.jit
def lift_state(state_std):
    """
    state_std: [x, y, theta, v, w] (5차원)
    Returns:   [x, y, cos, sin, v, w] (6차원 Koopman State)
    """
    x, y, theta, v, w = state_std
    return jnp.array([x, y, jnp.cos(theta), jnp.sin(theta), v, w])

@jax.jit
def step(state, control, dt):
    """
    state:   [x, y, c, s, v, w]
    control: [accel_v, accel_w] (가속도 입력)
    """
    x, y, c, s, v, w = state
    av, aw = control

    # 1. 속도 업데이트 (Euler Integration)
    next_v = v + av * dt
    next_w = w + aw * dt

    # 2. 위치 업데이트 (변경된 속도 기반)
    # 회전 행렬 계산 (Kinematics)
    rot_c = jnp.cos(next_w * dt)
    rot_s = jnp.sin(next_w * dt)

    next_c = c * rot_c - s * rot_s
    next_s = s * rot_c + c * rot_s
    
    # x, y 이동
    next_x = x + next_v * next_c * dt
    next_y = y + next_v * next_s * dt

    return jnp.array([next_x, next_y, next_c, next_s, next_v, next_w])

# =========================================================
# 2. QP Projector (Safety Filter + Terminal Constraints)
# =========================================================

class KoopmanQPProjector:
    def __init__(self, horizon, n_cp, dt, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.bspline = bspline_gen
        self.dim_var = n_cp * 2

        # [Solver] BatchedADMM (Custom Implementation)
        # rho: Penalty Parameter (2.0 ~ 5.0 권장)
        self.solver = BatchedADMM(n_vars=self.dim_var, rho=2.0, max_iter=50)

        # [Constraints] 입력은 이제 '가속도'입니다.
        # 선가속도: -1.0 ~ 1.0 m/s^2, 각가속도: -2.0 ~ 2.0 rad/s^2
        self.u_min = jnp.array([-1.0, -2.0]) 
        self.u_max = jnp.array([ 1.0,  2.0])

    @partial(jax.jit, static_argnums=(0,))
    def rollout_fn(self, coeffs, z0):
        # B-Spline으로 가속도 시퀀스 생성
        u_seq = self.bspline.get_sequence(coeffs)
        
        def step_fn(carry, u):
            z_next = step(carry, u, self.dt)
            return z_next, z_next # 전체 상태 반환 (6차원)
            
        _, traj = jax.lax.scan(step_fn, z0, u_seq)
        return traj

    @partial(jax.jit, static_argnums=(0,))
    def jac_fn(self, coeffs, z0):
        return jax.jacfwd(self.rollout_fn, argnums=0)(coeffs, z0)

    @partial(jax.jit, static_argnums=(0,))
    def project_single_sample(self, coeffs_noisy, z0, obs_pos, obs_r, target_pos):
        # 1. 궤적 및 자코비안 계산
        p_traj = self.rollout_fn(coeffs_noisy, z0)      # (H, 6) [x,y,c,s,v,w]
        J_tensor = self.jac_fn(coeffs_noisy, z0)        # (H, 6, N_CP, 2)

        # ---------------------------------------------------
        # Constraint 1: 장애물 회피 (위치 x, y만 사용)
        # ---------------------------------------------------
        pos_traj = p_traj[:, :2] # (H, 2)
        diff = pos_traj - obs_pos 
        dist_sq = jnp.sum(diff**2, axis=1)              
        dist_vals = jnp.sqrt(dist_sq) + 1e-6            
        normals = diff / dist_vals[:, None]             # (H, 2)

        # Linearization: - (Normal @ J) * delta <= - (req - dist)
        J_pos = J_tensor[:, :2, :, :] # (H, 2, N_CP, 2)
        J_flat = J_pos.reshape(self.H, 2, -1)
        
        A_obs = -jnp.einsum('td,tdv->tv', normals, J_flat)
        
        safe_margin = 0.3
        req_dist = obs_r + safe_margin
        b_obs = -(req_dist - dist_vals)
        l_obs = jnp.full_like(b_obs, -1e9) # One-sided

        # ---------------------------------------------------
        # Constraint 2: 입력 제한 (Box Constraint)
        # ---------------------------------------------------
        coeffs_flat = coeffs_noisy.reshape(-1)
        u_min_tiled = jnp.tile(self.u_min, self.N_cp)
        u_max_tiled = jnp.tile(self.u_max, self.N_cp)
        
        l_input = u_min_tiled - coeffs_flat
        u_input = u_max_tiled - coeffs_flat
        A_input = jnp.eye(self.dim_var)

        # ---------------------------------------------------
        # Constraint 3: [핵심] 종단 속도 0 강제 (Terminal Velocity)
        # ---------------------------------------------------
        # 마지막 상태의 v(4), w(5) 인덱스
        v_final = p_traj[-1, 4]
        w_final = p_traj[-1, 5]
        
        # Jacobian for v and w at terminal step
        J_v_end = J_tensor[-1, 4].reshape(1, -1)
        J_w_end = J_tensor[-1, 5].reshape(1, -1)
        
        # Eq: v_final + J * delta = 0  =>  J * delta = -v_final
        A_term = jnp.vstack([J_v_end, J_w_end])
        b_term = jnp.array([-v_final, -w_final])
        
        # 등식 제약이므로 l과 u를 아주 좁게 설정 (Soft Constraint 효과)
        tol = 1e-3
        l_term = b_term - tol
        u_term = b_term + tol

        # ---------------------------------------------------
        # 행렬 통합 (Stacking)
        # ---------------------------------------------------
        A = jnp.vstack([A_obs, A_input, A_term])
        l = jnp.concatenate([l_obs, l_input, l_term])
        u = jnp.concatenate([b_obs, u_input, u_term])

        P = jnp.eye(self.dim_var)
        q = jnp.zeros(self.dim_var)
        
        # ADMM Solve
        delta, final_state = self.solver.solve(P, q, A, l, u)

        safe_coeffs = coeffs_noisy + delta.reshape(self.N_cp, 2)
        
        return safe_coeffs, final_state

# =========================================================
# 3. Koopman MPPI (Updated Logic)
# =========================================================

class KoopmanMPPI:
    def __init__(self, horizon, n_cp, dt, n_samples, temperature, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.K = n_samples
        self.lambda_ = temperature
        self.projector = KoopmanQPProjector(horizon, n_cp, dt, bspline_gen)

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
        
        # 3. 부드러움 (가속도의 변화량 = Jerk 최소화)
        diffs = jnp.diff(coeffs, axis=0) 
        smoothness_cost = jnp.sum(diffs**2)
        
        # 4. [NEW] 에너지 및 종단 정지 비용
        # 에너지는 가속도(입력)의 크기
        energy_cost = jnp.sum(coeffs**2) 
        # 마지막 속도가 0이어야 함 (QP에서 강제하지만 Cost로도 유도)
        terminal_vel_cost = jnp.sum(vel_traj[-1]**2)

        total_cost = (
            30.0 * term_err + 
            1.0 * stage_err + 
            10.0 * obs_cost + 
            5.0 * smoothness_cost +
            0.1 * energy_cost +
            30.0 * terminal_vel_cost # 정지 중요도 높음
        )
        return total_cost

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, mean_coeffs, z0, target_pos, obs_pos, obs_r, prev_solver_state):
        # 1. Adaptive Noise (목표 근처에서 노이즈 감소)
        dist_to_goal = jnp.linalg.norm(z0[:2] - target_pos)
        sigma = jnp.where(dist_to_goal < 1.0, 0.2, 0.6) # 가속도 노이즈
        
        noise = jax.random.normal(key, (self.K, self.N_cp, 2)) * sigma
        raw_samples = mean_coeffs + noise
        
        # 2. Parallel Projection (Safety Filter)
        project_fn = jax.vmap(self.projector.project_single_sample, in_axes=(0, None, None, None, None))
        safe_samples, solver_states = project_fn(raw_samples, z0, obs_pos, obs_r, target_pos)
        
        # 3. Cost Evaluation & Masking
        cost_fn = jax.vmap(self.compute_cost, in_axes=(0, None, None, None, None))
        costs = cost_fn(safe_samples, z0, target_pos, obs_pos, obs_r)
        
        # 4. MPPI Update
        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.lambda_)
        
        weights_expanded = weights[:, None, None]
        new_mean = jnp.sum(weights_expanded * safe_samples, axis=0)
        
        alpha = 0.9 # Update rate
        final_mean = alpha * new_mean + (1 - alpha) * mean_coeffs
        
        # Warm Start Update (Best Sample's Dual Variables)
        best_idx = jnp.argmin(costs)
        # solver_states: tuple (x, z, y)
        next_solver_state = jax.tree_util.tree_map(lambda x: x[best_idx], solver_states)
        
        return final_mean, safe_samples, weights, next_solver_state

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
    mppi = KoopmanMPPI(HORIZON, N_CP, DT, N_SAMPLES, TEMP, bspline_gen)
    
    # [초기 상태] x, y, theta, v, w (5차원)
    start_pose = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    z_curr = lift_state(start_pose) # -> 6차원 [x,y,c,s,v,w]
    
    target_pos = jnp.array([5.0, 0.0])
    obs_pos = jnp.array([2.5, 0.0])
    obs_r = 0.8
    
    # 초기 제어 입력 (가속도 = 0)
    mean_coeffs = jnp.zeros((N_CP, 2))
    solver_state = None 
    
    key = jax.random.PRNGKey(0)
    traj_hist = [z_curr[:2]]
    
    log_solver_time = [] # ms

    print("Simulation Running...")
    
    plt.figure(figsize=(10, 6))
    
    for t in range(300):
        key, subkey = jax.random.split(key)

        jax.block_until_ready(mean_coeffs)
        t0 = time.time()

        # 1. MPPI Step
        mean_coeffs, safe_samples, weights, solver_state = mppi.step(
            subkey, mean_coeffs, z_curr, target_pos, obs_pos, obs_r, solver_state
        )
        
        # Block valid for timing accurate measurements
        jax.block_until_ready(mean_coeffs)
        t_end = time.time()

        solver_ms = (t_end - t0) * 1000.0
        log_solver_time.append(solver_ms)

        # 2. Apply Control
        u_seq = bspline_gen.get_sequence(mean_coeffs)
        u_curr = u_seq[0] # 현재 가속도 명령
        
        # [Goal Latching] 목표 근처 정밀 제어 및 정지
        dist_to_goal = jnp.linalg.norm(z_curr[:2] - target_pos)

        # 3. Simulate Dynamics
        z_curr = step(z_curr, u_curr, DT)
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
            plt.plot(mean_traj[:, 0], mean_traj[:, 1], 'b-', linewidth=3, label='Optimal Path')
            
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
    plt.plot(log_solver_time, label='Solver Time (ms)')
    plt.title(f"ADMM solver time per step")
    plt.xlabel("Simulation Step")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.show()
    # --- Final Stats Print ---
    print("\n" + "="*30)
    print(" [Simulation Result Summary]")
    print("="*30)
    print(f"Avg Solver Time: {np.mean(log_solver_time[3:]):.2f} ms")
    print(f"Max Solver Time: {np.max(log_solver_time[3:]):.2f} ms")
    print(f"Min Solver Time: {np.min(log_solver_time[3:]):.2f} ms")
    print(f"center solver time: {np.median(log_solver_time[3:]):.2f} ms")
    print("="*30)


if __name__ == "__main__":
    run()