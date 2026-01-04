import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
from Bspline import BSplineBasis
import time

@jax.jit
def lift_state(state_std):
    # 입력: [x, y, theta, v, w] (5차원) -> 이제 초기값에 속도도 필요함
    x, y, theta, v, w = state_std
    # 출력: [x, y, cos, sin, v, w] (6차원 Koopman State)
    return jnp.array([x, y, jnp.cos(theta), jnp.sin(theta), v, w])

@jax.jit
def step_(state, control, dt):
    # state: [x, y, c, s, v, w]
    # control: [accel_v, accel_w] (가속도 제어)
    x, y, c, s, v, w = state
    av, aw = control

    # 1. 속도 업데이트 (오일러 적분)
    next_v = v + av * dt
    next_w = w + aw * dt

    # 2. 위치/방향 업데이트 (새로운 속도 기반)
    # 회전 행렬 계산
    rot_c = jnp.cos(next_w * dt)
    rot_s = jnp.sin(next_w * dt)

    next_c = c * rot_c - s * rot_s
    next_s = s * rot_c + c * rot_s
    
    # 위치 업데이트
    next_x = x + next_v * next_c * dt
    next_y = y + next_v * next_s * dt

    return jnp.array([next_x, next_y, next_c, next_s, next_v, next_w])

# =========================================================
# 2. QP Projector with Limits & Warm Start (Fixed)
# =========================================================

class QPProjector:
    def __init__(self, horizon, n_cp, dt, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.bspline = bspline_gen
        
        # [최적화 유지] Solver 제한 완화 (전체 궤적을 넣으면 문제 크기가 커지므로 tol을 살짝 낮춤)
        self.qp = jaxopt.OSQP(tol=1e-2, maxiter=50)

        self.u_min = jnp.array([-1.0, -2.0])  # 감속, 우회전 가속 최대치
        self.u_max = jnp.array([ 1.0,  2.0])  # 가속, 좌회전 가속 최대치

    @partial(jax.jit, static_argnums=(0,))
    def rollout_fn(self, coeffs, z0):
        # (기존과 동일)
        u_seq = self.bspline.get_sequence(coeffs)
        def step_fn(carry, u):
            z_next = step_(carry, u, self.dt)
            return z_next, z_next[:2]
        _, pos_traj = jax.lax.scan(step_fn, z0, u_seq)
        return pos_traj

    # [수정] 전체 궤적에 대한 자코비안이 필요하므로 다시 jacfwd 사용
    # 입력(20) < 출력(30*2=60) 상황이므로 jacfwd가 효율적
    @partial(jax.jit, static_argnums=(0,))
    def jac_fn(self, coeffs, z0):
        return jax.jacfwd(self.rollout_fn, argnums=0)(coeffs, z0)

    @partial(jax.jit, static_argnums=(0,))
    def project_single_sample(self, coeffs_noisy, z0, obs_pos, obs_r, solver_init_state=None):
        # 1. 전체 궤적 및 자코비안 계산
        p_traj = self.rollout_fn(coeffs_noisy, z0)      # (H, 6) [x, y, c, s, v, w]
        J_tensor = self.jac_fn(coeffs_noisy, z0)        # (H, 6, N_CP, 2)

        # ---------------------------------------------------------
        # [추가됨] 2. Terminal State Constraint (v, w)
        # ---------------------------------------------------------
        # 마지막 상태의 v(4), w(5) 값 추출
        v_final = p_traj[-1, 4]
        w_final = p_traj[-1, 5]
        
        # Jacobian for v and w at terminal step
        # J_tensor shape: (H, 6, N_CP, 2) -> [-1, 4] select -> (N_CP, 2)
        # reshape(1, -1) -> (1, N_CP*2) : coeffs_flat과 차원 일치
        J_v_end = J_tensor[-1, 4].reshape(1, -1)
        J_w_end = J_tensor[-1, 5].reshape(1, -1)
        
        # Eq: v_final + J * delta = 0  =>  J * delta = -v_final
        A_term = jnp.vstack([J_v_end, J_w_end])       # (2, N_var)
        b_term = jnp.array([-v_final, -w_final])      # (2,)
        
        # 등식 제약이므로 l과 u를 아주 좁게 설정 (Soft Constraint 효과)
        tol = 1e-3
        l_term = b_term - tol
        u_term = b_term + tol

        # ---------------------------------------------------------
        # 3. Obstacle Avoidance Constraint (기존 로직)
        # ---------------------------------------------------------
        diff = p_traj[:, :2] - obs_pos 
        dist_sq = jnp.sum(diff**2, axis=1)
        dist_vals = jnp.sqrt(dist_sq) + 1e-6
        normals = diff / dist_vals[:, None]             # (H, 2)

        # J_flat: (H, 2, N_var) - 위치(0,1)에 대한 자코비안만 사용
        J_pos = J_tensor[:, :2, :, :].reshape(self.H, 2, -1)
        
        # A_obs_all: (H, N_var)
        A_obs_all = -jnp.einsum('td,tdv->tv', normals, J_pos)

        safe_margin = 0.3
        req_dist = obs_r + safe_margin
        b_obs_all = -(req_dist - dist_vals)

        # ---------------------------------------------------------
        # 4. Input Constraint (기존 로직)
        # ---------------------------------------------------------
        coeffs_flat = coeffs_noisy.reshape(-1)
        dim_var = self.N_cp * 2
        
        u_min_tiled = jnp.tile(self.u_min, self.N_cp)
        u_max_tiled = jnp.tile(self.u_max, self.N_cp)
        
        upper_bound = u_max_tiled - coeffs_flat
        lower_bound = u_min_tiled - coeffs_flat
        neg_lower_bound = -lower_bound

        # ---------------------------------------------------------
        # 5. 제약조건 병합 (Stacking)
        # OSQP format: G x <= h
        # ---------------------------------------------------------
        
        # (1) Obstacle: A_obs x <= b_obs
        # (2) Terminal Upper: A_term x <= u_term
        # (3) Terminal Lower: A_term x >= l_term  =>  -A_term x <= -l_term
        # (4) Input Max: I x <= upper
        # (5) Input Min: I x >= lower => -I x <= -lower
        
        G = jnp.vstack([
            A_obs_all,             # Obstacle
            A_term,                # Terminal (Upper bound)
            -A_term,               # Terminal (Lower bound)
            jnp.eye(dim_var),      # Input Max
            -jnp.eye(dim_var)      # Input Min
        ])

        h = jnp.concatenate([
            b_obs_all,             # Obstacle
            u_term,                # Terminal (Upper bound)
            -l_term,               # Terminal (Lower bound)
            upper_bound,           # Input Max
            neg_lower_bound        # Input Min
        ])

        # ---------------------------------------------------------
        # 6. QP 풀기
        # ---------------------------------------------------------
        P = jnp.eye(dim_var)
        q = jnp.zeros(dim_var)
        
        init_params = solver_init_state if solver_init_state is not None else None
        
        # params_ineq 만 사용 (등식 제약을 부등식 2개로 변환했으므로)
        sol = self.qp.run(
            init_params=init_params, 
            params_obj=(P, q), 
            params_ineq=(G, h)
        )

        delta = sol.params.primal.reshape(self.N_cp, 2)
        safe_coeffs = coeffs_noisy + delta
        
        # Cost 계산용 궤적 반환
        safe_traj = self.rollout_fn(safe_coeffs, z0)
        
        return safe_coeffs, safe_traj, sol.params

# =========================================================
# 3. Projected MPPI  (Updated Costs & Logic)
# =========================================================

class ProjectedMPPI:
    def __init__(self, horizon, n_cp, dt, n_samples, temperature, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.K = n_samples
        self.lambda_ = temperature
        self.projector = QPProjector(horizon, n_cp, dt, bspline_gen)

    @partial(jax.jit, static_argnums=(0,))
    def compute_cost(self, coeffs, z0, target_pos, obs_pos, obs_r):
        traj = self.projector.rollout_fn(coeffs, z0)
        
        # 1. 위치 오차 (기존)
        term_err = jnp.sum((traj[-1] - target_pos)**2)
        stage_err = jnp.sum(jnp.sum((traj - target_pos)**2, axis=1))
        
        # 2. 장애물 회피 (기존)
        dist = jnp.sqrt(jnp.sum((traj - obs_pos)**2, axis=1))
        obs_cost = jnp.sum(jnp.exp(-5.0 * (dist - obs_r - 0.2)))
        
        # 3. 입력 부드러움 (기존)
        diffs = jnp.diff(coeffs, axis=0) 
        smoothness_cost = jnp.sum(diffs**2)
        
        # (A) 전체 에너지 최소화 (불필요한 움직임 억제)
        energy_cost = jnp.sum(coeffs**2)
        
        # (B) 종단 속도 페널티 (마지막 순간에 멈추도록 강제)
        terminal_vel_cost = jnp.sum(coeffs[-3:]**2)

        # 가중치 조절
        total_cost = (
            30.0 * term_err + 
            1.0 * stage_err + 
            10.0 * obs_cost + 
            5.0 * smoothness_cost +
            0.1 * energy_cost +       # 움직임 최소화 (작게)
            30.0 * terminal_vel_cost  # 마지막에 멈추기 (크게!)
        )
        return total_cost

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, mean_coeffs, z0, target_pos, obs_pos, obs_r, prev_solver_params):
        # 1. Sampling
        dist_to_goal = jnp.linalg.norm(z0[:2] - target_pos)
        sigma = jnp.where(dist_to_goal < 1.0, 0.2, 0.6) # 가속도 노이즈
        
        noise = jax.random.normal(key, (self.K, self.N_cp, 2)) * sigma
        raw_samples = mean_coeffs + noise
        
        # 2. Project ALL Samples with Limits & Obstacles
        # vmap: project_single_sample(coeffs, z0, obs, r, solver_state)
        project_fn = jax.vmap(self.projector.project_single_sample, in_axes=(0, None, None, None, None))
        
        # prev_solver_params가 None이면 None이 전달됨
        safe_samples, _, solver_states = project_fn(raw_samples, z0, obs_pos, obs_r, prev_solver_params)
        
        # 3. Cost Evaluation
        cost_fn = jax.vmap(self.compute_cost, in_axes=(0, None, None, None, None))
        costs = cost_fn(safe_samples, z0, target_pos, obs_pos, obs_r)
        
        # 4. Weight Update
        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.lambda_)
        
        weights_expanded = weights[:, None, None]
        new_mean = jnp.sum(weights_expanded * safe_samples, axis=0)
        
        alpha = 0.9 # Smooth update
        final_mean = alpha * new_mean + (1 - alpha) * mean_coeffs
        
        # 다음 스텝 Warm Start를 위해, 가장 비용이 낮은(Best) 샘플의 솔버 상태를 반환
        best_idx = jnp.argmin(costs)
        next_solver_params = jax.tree.map(lambda x: x[best_idx], solver_states)
        
        return final_mean, safe_samples, weights, next_solver_params

# =========================================================
# 3. Main Simulation Loop
# =========================================================
def run():
    DT = 0.1
    HORIZON = 30
    N_CP = 10
    N_SAMPLES = 100 
    TEMP = 0.5
    
    bspline_gen = BSplineBasis(N_CP, HORIZON)
    mppi = ProjectedMPPI(HORIZON, N_CP, DT, N_SAMPLES, TEMP, bspline_gen)
    
   # [초기 상태] x, y, theta, v, w (5차원)
    start_pose = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    z_curr = lift_state(start_pose) # -> 6차원 [x,y,c,s,v,w]
    
    target_pos = jnp.array([5.0, 0.0])
    obs_pos = jnp.array([2.5, 0.0])
    obs_r = 0.8
    
    # Initial Guess (Straight forward)
    # v=1.0, w=0.0
    mean_coeffs = jnp.ones((N_CP, 2)) * jnp.array([1.0, 0.0])
    
    # Warm Start용 변수 초기화
    solver_params = None 
    
    key = jax.random.PRNGKey(0)
    traj_hist = [z_curr[:2]]

    log_solver_time = [] # ms
        
    print("Simulation Running ...")
    
    plt.figure(figsize=(10, 6))
    
    for t in range(500):
        key, subkey = jax.random.split(key)

        jax.block_until_ready(mean_coeffs)
        t0 = time.time()

        # MPPI Step with Warm Start passing
        mean_coeffs, safe_samples, weights, solver_params = mppi.step(
            subkey, mean_coeffs, z_curr, target_pos, obs_pos, obs_r, solver_params
        )
        
        # Block valid for timing accurate measurements
        jax.block_until_ready(mean_coeffs)
        t_end = time.time()

        solver_ms = (t_end - t0) * 1000.0
        log_solver_time.append(solver_ms)

        # Execute Control (First step of spline)
        u_seq = bspline_gen.get_sequence(mean_coeffs)
        u_curr = u_seq[0]
        
        dist_to_goal = jnp.linalg.norm(z_curr[:2] - target_pos)

        # Physics Update
        z_curr = step_(z_curr, u_curr, DT)
        traj_hist.append(z_curr[:2])

        # --- Visualization ---
        if t % 1 == 0:
            plt.clf()
            
            # Obstacle & Margin
            plt.gca().add_patch(plt.Circle(obs_pos, obs_r, color='r', alpha=0.5))
            plt.gca().add_patch(plt.Circle(obs_pos, obs_r + 0.3, color='r', fill=False, linestyle=':', label='QP Margin'))
            
            plt.plot(target_pos[0], target_pos[1], 'bx', markersize=10, label='Target')
            
            # Samples Visualization
            rollout_viz = jax.jit(jax.vmap(mppi.projector.rollout_fn, in_axes=(0, None)))
            top_idx = jnp.argsort(weights)[-20:] # Best 20 only
            top_samples = safe_samples[top_idx]
            top_trajs = rollout_viz(top_samples, z_curr)
            
            for k in range(len(top_samples)):
                alp = 0.2 + 0.8 * (k / len(top_samples))
                plt.plot(top_trajs[k, :, 0], top_trajs[k, :, 1], 'g-', alpha=alp)
            
            # Mean Trajectory
            mean_traj = mppi.projector.rollout_fn(mean_coeffs, z_curr)
            plt.plot(mean_traj[:, 0], mean_traj[:, 1], 'b-', linewidth=3, label='Mean Control')
            
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
            plt.legend()
            plt.pause(0.01)

        if dist_to_goal < 0.1:
            print("Goal Reached!")
            break
            
    plt.show()

    plt.figure()
    plt.plot(log_solver_time[3:], label='Solver Time (ms)')
    plt.title(f"OSQP solver time per step")
    plt.xlabel("Simulation Step")
    plt.ylabel("Time (ms)")
    plt.legend()
    plt.show()
    # --- Final Stats Print ---
    print("\n" + "="*30)
    print(" [Simulation Result Summary (OSQP)]")
    print("="*30)
    print(f"Avg Solver Time: {np.mean(log_solver_time[3:]):.2f} ms")
    print(f"Max Solver Time: {np.max(log_solver_time[3:]):.2f} ms")
    print(f"Min Solver Time: {np.min(log_solver_time[3:]):.2f} ms")
    print(f"center solver time: {np.median(log_solver_time[3:]):.2f} ms")
    print("="*30)

if __name__ == "__main__":
    run()