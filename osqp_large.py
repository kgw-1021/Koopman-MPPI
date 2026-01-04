import jax
import jax.numpy as jnp
import jaxopt
import matplotlib.pyplot as plt
from functools import partial
import numpy as np
import time

# =========================================================
# 0. Helper & Dynamics (기존 osqp.py 내용 기반)
# =========================================================

# B-Spline 클래스 (파일이 없을 경우를 대비해 내장)
class BSplineBasis:
    def __init__(self, n_cp, horizon, dt=0.05):
        self.n_cp = n_cp
        self.horizon = horizon
        self.dt = dt
        # 간단한 균등 B-Spline 기저함수 행렬 생성 (Linear Approximation for simple viz)
        # 실제로는 scipy.interpolate.BSpline 등을 사용하여 정확히 구현 권장
        t = jnp.linspace(0, 1, horizon)
        self.basis_mat = self._make_basis(t, n_cp)

    def _make_basis(self, t, n_cp):
        # RBF를 이용한 유사 B-Spline 기저 (예시용)
        basis = []
        for i in range(n_cp):
            c = i / (n_cp - 1)
            vals = jnp.exp(-0.5 * ((t - c) / 0.15)**2)
            basis.append(vals)
        basis = jnp.stack(basis, axis=1)
        # Normalize
        basis = basis / (jnp.sum(basis, axis=1, keepdims=True) + 1e-6)
        return basis # (H, N_cp)

@jax.jit
def lift_state(state_std):
    # [x, y, theta, v, w] -> [x, y, cos, sin, v, w]
    x, y, theta, v, w = state_std
    return jnp.array([x, y, jnp.cos(theta), jnp.sin(theta), v, w])

@jax.jit
def step_dynamics(state, control, dt):
    # Unicycle Dynamics
    x, y, c, s, v, w = state
    av, aw = control # 가속도 제어

    next_v = v + av * dt
    next_w = w + aw * dt
    
    # 방향 업데이트
    theta = jnp.arctan2(s, c)
    next_theta = theta + next_w * dt
    next_c = jnp.cos(next_theta)
    next_s = jnp.sin(next_theta)

    # 위치 업데이트
    next_x = x + next_v * c * dt
    next_y = y + next_v * s * dt

    return jnp.array([next_x, next_y, next_c, next_s, next_v, next_w])

# =========================================================
# 1. Stacked Projector (핵심 구현)
# =========================================================

class QPProjector:
    """
    모든 MPPI 샘플(K개)을 하나의 거대한 QP로 묶어서(Stacking) OSQP로 풉니다.
    구조: Block Diagonal Matrix 형태의 제약조건을 Implicit하게 처리함.
    """
    def __init__(self, horizon, n_cp, dt, bspline_basis, n_samples):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.basis = bspline_basis.basis_mat  # (H, N_cp)
        self.K = n_samples
        
        # 제어 입력: [accel_v, accel_w]
        self.dim_u = 2 
        self.n_vars_per_sample = self.N_cp * self.dim_u
        self.total_vars = self.K * self.n_vars_per_sample

        # -----------------------------------------------------------
        # [핵심] OSQP Solver 설정 (Implicit Block Diagonal)
        # -----------------------------------------------------------
        
        def matvec_A(A_data, x_flat):
            """
            A_data: (K, m_local, n_local) - 각 샘플별로 쌓인 제약 행렬
            x_flat: (K * n_local, )       - 전체 최적화 변수 (delta)
            """
            # [수정] params_eq가 None으로 들어올 때의 예외 처리
            if A_data is None:
                return jnp.zeros((0,))

            # 1. 벡터를 샘플별로 쪼갬
            x_reshaped = x_flat.reshape(self.K, self.n_vars_per_sample)
            
            # 2. 배치 행렬 곱 (Block Diagonal 곱셈과 동일)
            # einsum: 각 샘플 k에 대해 A_data[k] @ x_reshaped[k] 수행
            Ax_batch = jnp.einsum('kmn,kn->km', A_data, x_reshaped)
            
            # 3. 다시 1줄로 폄 (OSQP 내부용)
            return Ax_batch.ravel()

        # matvec_Q: Px 계산 (Objective: min ||delta||^2)
        # P는 Identity이므로 입력 그대로 반환
        def matvec_Q(Q_params, x_flat):
            return x_flat 

        # OSQP 인스턴스 생성
        self.osqp = jaxopt.OSQP(matvec_A=matvec_A, matvec_Q=matvec_Q, tol=1e-2, maxiter=50)

    # --- Dynamics Rollout Helper ---
    def rollout_trajectory(self, coeffs_flat, init_state_lifted):
        coeffs = coeffs_flat.reshape(self.N_cp, self.dim_u)
        # B-Spline Basis로 전체 제어 입력(가속도) 생성: (H, 2)
        u_traj = self.basis @ coeffs 
        
        def scan_fn(state, u):
            next_state = step_dynamics(state, u, self.dt)
            return next_state, next_state[:2] # (x,y) 위치만 저장

        _, pos_traj = jax.lax.scan(scan_fn, init_state_lifted, u_traj)
        return pos_traj # (H, 2)
        
    # --- Visualization Helper (시각화용 함수 추가) ---
    def rollout_fn(self, coeffs, init_state):
        # MPPI 시각화 루프에서 사용됨
        coeffs_flat = coeffs.reshape(-1)
        return self.rollout_trajectory(coeffs_flat, init_state)

    # --- Constraint Generation (Linearization) ---
    @partial(jax.jit, static_argnums=(0,))
    def get_batch_constraints(self, samples_flat, init_state, obs_pos, obs_r):
        """
        모든 샘플에 대해 선형화된 제약조건(A, l, u) 생성
        samples_flat: (K, N_cp*2)
        """
        
        # 1. Jacobian 계산을 위한 함수 (Input: coeffs -> Output: Trajectory)
        def traj_fn(c_flat):
            return self.rollout_trajectory(c_flat, init_state)

        def val_and_jac_fn(x):
             return traj_fn(x), jax.jacfwd(traj_fn)(x)
             
        pos_batch, jac_batch = jax.vmap(val_and_jac_fn)(samples_flat)

        # 3. 장애물 회피 제약 (SCP Linearization)
        # 조건: ||p - p_obs|| >= R  (Non-convex)
        # 선형화: -n^T * J * delta <= -(R_safe - dist_curr)
        
        diff = pos_batch - obs_pos 
        dist = jnp.linalg.norm(diff, axis=-1, keepdims=True) # (K, H, 1)
        safe_dist = obs_r + 0.5  # 여유 마진 포함

        # Normal vector
        normals = diff / (dist + 1e-6) # (K, H, 2)
        
        # A_obs: (K, H, N_vars)
        # -n^T * J
        A_obs = -jnp.einsum('ktd,ktdn->ktn', normals, jac_batch)
        
        # b_obs: (K, H)
        # dist < safe_dist 이면 우변이 음수 -> delta가 강제로 이동해야 함
        b_obs = -(safe_dist - dist.squeeze(-1))
        
        # 4. 입력 범위 제약 (Box Constraint)
        # u_min <= u_nom + delta <= u_max
        # 여기서는 Coeffs 자체의 가속도 크기를 제한한다고 가정 (-2 ~ 2)
        u_limit = 4.0 
        
        # A_inp = Identity (K, N_vars, N_vars)
        A_inp = jnp.tile(jnp.eye(self.n_vars_per_sample), (self.K, 1, 1))
        
        l_inp = -u_limit - samples_flat
        u_inp =  u_limit - samples_flat

        # 5. 행렬 합치기 (Stacking) -> 하나의 큰 A 행렬(논리적) 구성
        # A_local: (K, H + N_vars, N_vars)
        A_local = jnp.concatenate([A_obs, A_inp], axis=1)
        
        # l_local, u_local: (K, H + N_vars)
        # 장애물은 한쪽 제약(-inf ~ b), 입력은 양쪽 제약(l ~ u)
        l_obs = jnp.full_like(b_obs, -1e9)
        l_local = jnp.concatenate([l_obs, l_inp], axis=1)
        u_local = jnp.concatenate([b_obs, u_inp], axis=1)
        
        return A_local, l_local, u_local

    @partial(jax.jit, static_argnums=(0,))
    def project_batch(self, samples, init_state, obs_pos, obs_r, init_params=None):
        """
        Main Interface: K개의 샘플을 받아 안전한 샘플로 투영
        """
        # Flatten input: (K, N_cp, 2) -> (K, N_vars)
        samples_flat = samples.reshape(self.K, -1)
        
        # 1. 제약조건 생성
        A_batch, l_batch, u_batch = self.get_batch_constraints(samples_flat, init_state, obs_pos, obs_r)

        # 2. Double-sided 제약을 Single-sided (Ax <= b) 2개로 분리
        # (l <= Ax <= u)  <==>  (Ax <= u) AND (-Ax <= -l)
        A_dual = jnp.concatenate([A_batch, -A_batch], axis=1) # Shape: (K, 2m, n)
        b_dual = jnp.concatenate([u_batch, -l_batch], axis=1) # Shape: (K, 2m)
        
        # [수정] 3D 배치를 하나의 큰 2D Block Diagonal Matrix로 변환
        # JAXopt는 부등식 제약조건에 대해 3D 배치를 처리하지 못하므로, (K*2m, K*n) 형태로 만듭니다.
        
        # 3차원 배열을 K개의 2차원 배열 리스트로 분리 (JIT 컴파일 시 Loop Unrolling 됨)
        A_list = [A_dual[i] for i in range(self.K)]
        
        # 블록 대각 행렬 생성 (약 100개 샘플 정도는 Dense Matrix로 합쳐도 메모리 문제 없음)
        A_giant = jax.scipy.linalg.block_diag(*A_list)
        
        # 벡터 펼치기
        b_giant = b_dual.ravel()
        q_giant = jnp.zeros(self.total_vars)

        # 3. OSQP 실행
        # A_giant: (K*2m, K*n), b_giant: (K*2m) -> 차원이 일치하여 dot 연산 가능
        sol = self.osqp.run(
            init_params=init_params,
            params_obj=(None, q_giant),
            params_ineq=(A_giant, b_giant) 
        )
        
        # 4. 결과 복원 (Delta + Samples)
        delta_giant = sol.params.primal
        delta_batch = delta_giant.reshape(self.K, self.N_cp, self.dim_u)
        
        safe_samples = samples + delta_batch
        return safe_samples, sol.params
# =========================================================
# 2. Main Simulation Class
# =========================================================

class ProjectedMPPI:
    def __init__(self, horizon, n_cp, dt, n_samples, temperature, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.K = n_samples
        self.lambda_ = temperature
        self.projector = QPProjector(self.H, self.N_cp, self.dt, bspline_gen, self.K)
        
        self.bspline = BSplineBasis(self.N_cp, self.H, self.dt)
        
        # [NEW] Stacked Projector 초기화
        self.projector = QPProjector(
            self.H, self.N_cp, self.dt, self.bspline, self.K
        )
        self.solver_state = None

    def compute_cost(self, samples, target, obs_pos):
        # samples: (K, N_cp, 2)
        # Rollout trajectories
        def single_rollout(coeffs):
            coeffs = coeffs.reshape(self.N_cp, 2)
            u_traj = self.bspline.basis_mat @ coeffs
            # Simple rollout for cost (using scan)
            def scan_fn(state, u):
                # Init state 0 for relative rollout or use actual state
                # Here assuming simple cost calculation from 0
                ns = step_dynamics(state, u, self.dt)
                return ns, ns[:2]
            
            # Start from 0,0 for shape cost (Actual MPPI uses current state)
            # 여기서는 편의상 현재 위치(0,0) 가정 혹은 별도 처리
            _, pos = jax.lax.scan(scan_fn, jnp.zeros(6), u_traj)
            return pos

        # 배치 롤아웃은 생략하고, 간단히 목표지점 거리 + 입력 크기 비용 계산
        # (실제로는 Projector가 안전을 보장하므로 Cost는 가이던스 역할만 함)
        coeffs_flat = samples.reshape(self.K, -1)
        cost_input = jnp.sum(coeffs_flat**2, axis=1) * 0.01
        
        # Dummy Goal Cost (단순화를 위해 입력 방향성만 봄)
        # 실제 구현시엔 현재 상태에서 롤아웃 필요
        return cost_input

    def step(self, key, mean_coeffs, z_curr, target, obs_pos, obs_r, prev_solver_params):
        # 1. Sampling
        dist_to_goal = jnp.linalg.norm(z_curr[:2] - target)
        sigma = jnp.where(dist_to_goal < 1.0, 0.2, 0.6) # 가속도 노이즈
        
        noise = jax.random.normal(key, (self.K, self.N_cp, 2)) * sigma
        samples_noisy = mean_coeffs + noise
        
        # 2. [NEW] Stacked Batch Projection
        # 모든 샘플을 한 번에 OSQP로 투영
        safe_samples, solver_state = self.projector.project_batch(
            samples_noisy, z_curr, obs_pos, obs_r, prev_solver_params
        )
        self.solver_state = solver_state # Warm start for next step

        # 3. Cost & Weighting
        # (여기서는 투영된 샘플들 간의 거리가 비슷하므로 단순화)
        # 실제로는 Trajectory Cost 계산 필요. 여기서는 시각화용 데모이므로 생략.
        costs = jnp.sum((safe_samples - mean_coeffs)**2, axis=(1,2)) # Stay close to mean logic
        
        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.temp)
        
        # 4. Update Mean
        new_mean = jnp.sum(weights[:, None, None] * safe_samples, axis=0)
        
        # Moving Average
        mean_coeffs = 0.9 * new_mean + 0.1 * mean_coeffs
        
        return mean_coeffs, safe_samples, weights, solver_state

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
        z_curr = step_dynamics(z_curr, u_curr, DT)
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
    plt.plot(log_solver_time, label='Solver Time (ms)')
    plt.title(f"OSQP solver time per step")
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