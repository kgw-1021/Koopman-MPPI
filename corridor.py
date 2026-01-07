import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functools import partial
import numpy as np
from Bspline import BSplineBasis
from optimize import BatchedADMM
import os
import shutil   
import io
import imageio.v2 as imageio

def compute_free_energy(costs, lambda_):
    min_cost = jnp.min(costs)
    # Log-Sum-Exp Trick으로 안정적인 계산
    # F = min_cost - lambda * log( mean( exp( -(c - min)/lambda ) ) )
    weights_unnorm = jnp.exp(-(costs - min_cost) / lambda_)
    return min_cost - lambda_ * jnp.log(jnp.mean(weights_unnorm))

# =========================================================
# 1. System Dynamics (가속도 제어)
# =========================================================
@jax.jit
def lift_state(state_std):
    # [x, y, theta, v, w] -> [x, y, c, s, v, w]
    x, y, theta, v, w = state_std
    return jnp.array([x, y, jnp.cos(theta), jnp.sin(theta), v, w])

@jax.jit
def step_(state, control, dt):
    x, y, c, s, v, w = state
    av, aw = control
    next_v = v + av * dt
    next_w = w + aw * dt
    rot_c = jnp.cos(next_w * dt)
    rot_s = jnp.sin(next_w * dt)
    next_c = c * rot_c - s * rot_s
    next_s = s * rot_c + c * rot_s
    next_x = x + next_v * next_c * dt
    next_y = y + next_v * next_s * dt
    return jnp.array([next_x, next_y, next_c, next_s, next_v, next_w])

# =========================================================
# 2. QP Projector (Dynamic Gap Constraints)
# =========================================================

class QPProjector:
    def __init__(self, horizon, n_cp, dt, bspline_gen):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.bspline = bspline_gen
        self.dim_var = n_cp * 2
        # 선형 제약조건(Box constraints)이므로 rho를 5.0으로 강하게 설정
        self.solver = BatchedADMM(n_vars=self.dim_var, rho=5.0, max_iter=50)
        
        self.u_min = jnp.array([-1.5, -2.5]) 
        self.u_max = jnp.array([ 1.5,  2.5])

    @partial(jax.jit, static_argnums=(0,))
    def rollout_fn(self, coeffs, z0):
        u_seq = self.bspline.get_sequence(coeffs)
        def step_fn(carry, u):
            z_next = step_(carry, u, self.dt)
            return z_next, z_next
        _, traj = jax.lax.scan(step_fn, z0, u_seq)
        return traj

    @partial(jax.jit, static_argnums=(0,))
    def jac_fn(self, coeffs, z0):
        return jax.jacfwd(self.rollout_fn, argnums=0)(coeffs, z0)
    
    def get_corridor_bounds(self, x_pos):
        # x가 4.0 ~ 7.0 사이면 0.6 (Gap Half Width), 아니면 3.0 (Wide)
        # Smooth한 전환을 위해 sigmoid 대신 simple where 사용 (ADMM은 강건하므로 괜찮음)
        is_in_gap = (x_pos >= 3.0) & (x_pos <= 9.0)
        
        # Gap Width: 0.4m (Half: 0.2)
        # Room Width: 6.0m (Half: 3.0)
        bound = jnp.where(is_in_gap, 0.2, 3.0)
        return bound

    @partial(jax.jit, static_argnums=(0,))
    def project_single_sample(self, coeffs_noisy, z0, target_pos):

        p_traj = self.rollout_fn(coeffs_noisy, z0)      # (H, 6)
        J_tensor = self.jac_fn(coeffs_noisy, z0)        # (H, 6, N_CP, 2)
        
        pos_x = p_traj[:, 0] # (H,)
        pos_y = p_traj[:, 1] # (H,)
        
        # 1. 위치별 허용 범위(Bound) 계산
        y_bounds = self.get_corridor_bounds(pos_x) # (H,)
        
        # 2. Linearize Constraints
        # Upper Wall:  1*y <= bound  =>  1*(y + J*d) <= bound
        # Lower Wall: -1*y <= bound  => -1*(y + J*d) <= bound (즉 y >= -bound)
        
        # Jacobian for y (index 1)
        J_y = J_tensor[:, 1, :, :]      # (H, N_CP, 2)
        J_flat = J_y.reshape(self.H, -1) # (H, N_vars)
        
        # A matrix construction
        # Upper constraints (Row 0 ~ H-1)
        A_upper = J_flat
        b_upper = y_bounds - pos_y # upper_bound - current_y
        
        # Lower constraints (Row H ~ 2H-1) -> multiplied by -1
        A_lower = -J_flat
        b_lower = y_bounds + pos_y # bound - (-current_y) = bound + current_y
        
        # Upper/Lower Bounds for ADMM (Inequality: -inf <= Ax <= b)
        l_ineq = jnp.full(2 * self.H, -1e9)
        u_ineq = jnp.concatenate([b_upper, b_lower])
        A_ineq = jnp.vstack([A_upper, A_lower])

        # 3. Input Constraints
        coeffs_flat = coeffs_noisy.reshape(-1)
        l_input = jnp.tile(self.u_min, self.N_cp) - coeffs_flat
        u_input = jnp.tile(self.u_max, self.N_cp) - coeffs_flat
        A_input = jnp.eye(self.dim_var)

        # 4. Terminal Velocity Constraint (v, w = 0)
        v_final = p_traj[-1, 4]
        w_final = p_traj[-1, 5]
        J_v_end = J_tensor[-1, 4].reshape(1, -1)
        J_w_end = J_tensor[-1, 5].reshape(1, -1)
        
        A_term = jnp.vstack([J_v_end, J_w_end])
        b_term = jnp.array([-v_final, -w_final])
        l_term = b_term - 1e-3
        u_term = b_term + 1e-3

        # 5. Stack All
        A = jnp.vstack([A_ineq, A_input, A_term])
        l = jnp.concatenate([l_ineq, l_input, l_term])
        u = jnp.concatenate([u_ineq, u_input, u_term])
        
        P = jnp.eye(self.dim_var)
        q = jnp.zeros(self.dim_var)
        
        delta, final_state = self.solver.solve(P, q, A, l, u)
        return coeffs_noisy + delta.reshape(self.N_cp, 2), final_state

# =========================================================
# 3. MPPI Controller
# =========================================================

class ProjectedMPPI:
    def __init__(self, horizon, n_cp, dt, n_samples, temperature, bspline_gen):
        self.projector = QPProjector(horizon, n_cp, dt, bspline_gen)
        self.K = n_samples
        self.lambda_ = temperature
        self.N_cp = n_cp

    @partial(jax.jit, static_argnums=(0,))
    def compute_cost(self, coeffs, z0, target_pos):
        traj = self.projector.rollout_fn(coeffs, z0)
        pos_x = traj[:, 0]
        pos_y = traj[:, 1]
        
        # 1. Target Cost
        term_err = jnp.sum((traj[-1, :2] - target_pos)**2)
        
        # 2. Corridor Violation Cost (Soft Constraint)
        bounds = self.projector.get_corridor_bounds(pos_x)
        # 범위를 벗어나면 비용 폭증 (Safety Margin 0.1m 고려)
        violation = jnp.maximum(jnp.abs(pos_y) - (bounds - 0.1), 0.0)
        wall_cost = jnp.sum(jnp.exp(10.0 * violation) - 1.0)
        
        # 3. Control & Smoothness
        diffs = jnp.diff(coeffs, axis=0)
        smooth_cost = jnp.sum(diffs**2)
        stop_cost = jnp.sum(traj[-1, 4:]**2)

        return 50.0*term_err + 10.0*wall_cost + 5.0*smooth_cost + 30.0*stop_cost

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, mean_coeffs, z0, target_pos, prev_solver_state):
        dist_to_goal = jnp.linalg.norm(z0[:2] - target_pos)
        sigma = jnp.where(dist_to_goal < 1.0, 0.2, 0.6) # 가속도 노이즈
        
        noise = jax.random.normal(key, (self.K, self.N_cp, 2)) * sigma
        raw_samples = mean_coeffs + noise
        
        # 1. Projection (Narrow Gap 통과 유도)
        project_fn = jax.vmap(self.projector.project_single_sample, in_axes=(0, None, None))
        safe_samples, solver_states = project_fn(raw_samples, z0, target_pos)
        
        # 2. Cost Evaluation
        cost_fn = jax.vmap(self.compute_cost, in_axes=(0, None, None))
        costs = cost_fn(safe_samples, z0, target_pos)

        # 3. Weight Update
        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.lambda_)
        
        weights_expanded = weights[:, None, None]
        new_mean = jnp.sum(weights_expanded * safe_samples, axis=0)
        
        final_mean = 0.8 * new_mean + 0.2 * mean_coeffs
        
        # Solver Warm Start
        best_idx = jnp.argmin(costs)
        next_solver_state = jax.tree.map(lambda x: x[best_idx], solver_states)
        
        return final_mean, safe_samples, weights, costs, next_solver_state

# =========================================================
# 4. Simulation: Narrow Corridor
# =========================================================

def run(save_gif=True, gif_filename="our_result.gif"):
    DT = 0.1
    HORIZON = 30
    N_CP = 10
    N_SAMPLES = 100  
    TEMP = 0.5    
    
    bspline_gen = BSplineBasis(N_CP, HORIZON)
    mppi = ProjectedMPPI(HORIZON, N_CP, DT, N_SAMPLES, TEMP, bspline_gen)
    
    # --- Scenario Setup ---
    start_pose = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    z_curr = lift_state(start_pose)
    target_pos = jnp.array([10.0, 0.0])
    
    mean_coeffs = jnp.zeros((N_CP, 2))
    solver_state = None

    key = jax.random.PRNGKey(0)
    traj_hist = [z_curr[:2]]
    
    frames = []
    energy_history = []


    print("Simulation Running...")
    
    plt.figure(figsize=(12, 5))
    
    for t in range(300):
        key, subkey = jax.random.split(key)
        
        # Step
        mean_coeffs, safe_samples, weights, costs, solver_state = mppi.step(
            subkey, mean_coeffs, z_curr, target_pos, solver_state
        )
        
        free_energy = compute_free_energy(costs, mppi.lambda_)
        energy_history.append(free_energy)

        # Control
        u_seq = bspline_gen.get_sequence(mean_coeffs)
        u_curr = u_seq[0]
        
        # Dynamics
        z_curr = step_(z_curr, u_curr, DT)
        traj_hist.append(z_curr[:2])
        
        # Check Goal
        dist_to_goal = jnp.linalg.norm(z_curr[:2] - target_pos)

        # Visualization
        if t % 1 == 0:
            plt.clf()
            ax = plt.gca()
            
            # Draw Obstacles
            ax.add_patch(Rectangle((3, 0.2), 6, 2.8, color='k', alpha=0.6, label='Obstacle'))
            ax.add_patch(Rectangle((3, -3.0), 6, 2.8, color='k', alpha=0.6))
            
            # Draw Safe Bounds (Dotted Line)
            plt.plot([0, 3, 3, 9, 9, 12], [3, 3, 0.2, 0.2, 3, 3], 'k--', alpha=0.3)
            plt.plot([0, 3, 3, 9, 9, 12], [-3, -3, -0.2, -0.2, -3, -3], 'k--', alpha=0.3)
            
            # Samples (Top 20)
            rollout_viz = jax.jit(jax.vmap(mppi.projector.rollout_fn, in_axes=(0, None)))
            top_idx = jnp.argsort(weights)[-20:]
            top_samples = safe_samples[top_idx]
            top_trajs = rollout_viz(top_samples, z_curr)
            
            for k in range(len(top_samples)):
                alp = 0.3 + 0.7 * (k / len(top_samples))
                plt.plot(top_trajs[k, :, 0], top_trajs[k, :, 1], 'g-', alpha=alp)
            
            # Mean Path
            mean_traj = mppi.projector.rollout_fn(mean_coeffs, z_curr)
            plt.plot(mean_traj[:, 0], mean_traj[:, 1], 'b-', linewidth=3, label='Optimal Path')
            
            # History
            hist = np.array(traj_hist)
            plt.plot(hist[:, 0], hist[:, 1], 'r--', linewidth=2, label='Driven')
            
            # Target
            plt.plot(target_pos[0], target_pos[1], 'bx', markersize=12, markeredgewidth=3)
            
            plt.title(f"Step {t} | Dist: {dist_to_goal:.2f}m | V:{z_curr[4]:.2f}")
            plt.xlim(-1, 12)
            plt.ylim(-3.5, 3.5)
            plt.legend(loc='upper right')

            if save_gif:
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=180)
                buf.seek(0)
                frames.append(imageio.imread(buf))
                buf.close()
            else:
                plt.pause(0.01)

        if dist_to_goal < 0.1:
            print("Goal Reached!")
            break

    # --- Loop 종료 후 GIF 생성 ---
    if save_gif and len(frames) > 0:
        imageio.mimsave(gif_filename, frames, fps=8, loop=0)
        print("Done!")
    else: 
        plt.show()
    avg_energy = np.mean(energy_history)
    std_energy = np.std(energy_history)
    print(f"\n[Result] Averge Energy : {avg_energy:.2f}, Std Energy : {std_energy:.2f}")

if __name__ == "__main__":
    run(save_gif=False, gif_filename="our_result.gif")