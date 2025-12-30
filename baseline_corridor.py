import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from functools import partial
import numpy as np
import os
import shutil
import io
import imageio.v2 as imageio

# =========================================================
# 1. Utils
# =========================================================
def get_bspline_matrix(n_cp, horizon, degree=3):
    m = n_cp + degree + 1
    t = jnp.linspace(0, 1, horizon)
    knots = jnp.concatenate([
        jnp.zeros(degree),
        jnp.linspace(0, 1, n_cp - degree + 1),
        jnp.ones(degree)
    ])

    def basis(i, k, t):
        if k == 0:
            return jnp.where((t >= knots[i]) & (t < knots[i+1]), 1.0, 0.0)
        denom1 = knots[i+k] - knots[i]
        term1 = 0.0 if denom1 <= 0 else ((t - knots[i]) / denom1) * basis(i, k-1, t)
        denom2 = knots[i+k+1] - knots[i+1]
        term2 = 0.0 if denom2 <= 0 else ((knots[i+k+1] - t) / denom2) * basis(i+1, k-1, t)
        return term1 + term2

    M_list = []
    t_safe = jnp.clip(t, 0, 1.0 - 1e-6)
    for i in range(n_cp):
        M_list.append(basis(i, degree, t_safe))
    return jnp.stack(M_list, axis=1)

def compute_free_energy(costs, lambda_):
    min_cost = jnp.min(costs)
    # Log-Sum-Exp Trick으로 안정적인 계산
    # F = min_cost - lambda * log( mean( exp( -(c - min)/lambda ) ) )
    weights_unnorm = jnp.exp(-(costs - min_cost) / lambda_)
    return min_cost - lambda_ * jnp.log(jnp.mean(weights_unnorm))

# =========================================================
# 2. System Dynamics (Unicycle Model)
# =========================================================
@jax.jit
def step_dynamics(state, control, dt):
    x, y, theta, v, w = state
    a_v, a_w = control
    
    next_x = x + v * jnp.cos(theta) * dt
    next_y = y + v * jnp.sin(theta) * dt
    next_theta = theta + w * dt
    next_v = v + a_v * dt
    next_w = w + a_w * dt
    
    return jnp.array([next_x, next_y, next_theta, next_v, next_w])

# =========================================================
# 3. B-Spline MPPI Controller
# =========================================================

class BSplineMPPI:
    def __init__(self, horizon, n_cp, dt, n_samples, temperature, u_min, u_max):
        self.H = horizon
        self.N_cp = n_cp
        self.dt = dt
        self.K = n_samples
        self.lambda_ = temperature
        self.u_min = u_min
        self.u_max = u_max
        self.M = get_bspline_matrix(n_cp, horizon)

    @partial(jax.jit, static_argnums=(0,))
    def get_control_sequence(self, coeffs):
        return self.M @ coeffs

    @partial(jax.jit, static_argnums=(0,))
    def rollout(self, coeffs, state_init):
        u_seq = self.get_control_sequence(coeffs)
        def scan_fn(carry, u):
            next_s = step_dynamics(carry, u, self.dt)
            return next_s, next_s
        _, traj = jax.lax.scan(scan_fn, state_init, u_seq)
        return traj

    @partial(jax.jit, static_argnums=(0,))
    def compute_cost(self, coeffs, state_init, target_pos):
        # 1. Input Constraint Cost
        violation_lo = jnp.maximum(self.u_min - coeffs, 0.0)
        violation_hi = jnp.maximum(coeffs - self.u_max, 0.0)
        input_cost = jnp.sum(violation_lo**2 + violation_hi**2) * 500.0
        
        # 2. Clamped Rollout
        coeffs_clamped = jnp.clip(coeffs, self.u_min, self.u_max)
        traj = self.rollout(coeffs_clamped, state_init)
        
        pos_x = traj[:, 0]
        pos_y = traj[:, 1]
        
        # 3. Target Cost
        dist_sq = jnp.sum((traj[:, :2] - target_pos)**2, axis=1)
        time_weight = jnp.linspace(1.0, 5.0, self.H)
        goal_cost = jnp.sum(dist_sq * time_weight)
        # 4. Obstacle Cost
        is_in_gap = (pos_x >= 3.0) & (pos_x <= 9.0)
        
        # Gap Half Width: 0.2m
        safe_margin = 0.2 
        wide_margin = 3.0
        
        # Margin을 넘었을 때의 위반 정도 (양수면 충돌)
        violation_gap = jnp.where(is_in_gap, jnp.abs(pos_y) - safe_margin, -1.0)
        violation_wide = jnp.where(~is_in_gap, jnp.abs(pos_y) - wide_margin, -1.0)
        
        # Penalty 강화:
        obs_cost = jnp.sum(jnp.exp(50.0 * violation_gap) + jnp.exp(10.0 * violation_wide))
        
        # 5. Smoothness & Terminal
        diff_cost = jnp.sum(jnp.diff(coeffs_clamped, axis=0)**2) * 5.0
        term_vel_cost = (traj[-1, 3]**2 + traj[-1, 4]**2) * 100.0
        
        # obs_cost 가중치도 5.0 -> 20.0으로 상향하여 충돌 회피 최우선시
        return goal_cost + 20.0 * obs_cost + diff_cost + term_vel_cost + input_cost

    @partial(jax.jit, static_argnums=(0,))
    def step(self, key, mean_coeffs, state_init, target_pos):
        # 1. Sample Generation with Adaptive Noise
        dist_to_goal = jnp.linalg.norm(state_init[:2] - target_pos)
        
        # 거리가 1.0 미만이면 노이즈를 0.2로 줄이고, 아니면 0.6 사용
        sigma = jnp.where(dist_to_goal < 1.0, 0.2, 0.6) 
        
        # (K, N_cp, 2) 크기의 노이즈 생성 후 sigma 적용
        noise = jax.random.normal(key, (self.K, self.N_cp, 2)) * sigma
        
        samples = mean_coeffs + noise
        
        # 2. Cost Evaluation
        costs = jax.vmap(self.compute_cost, in_axes=(0, None, None))(samples, state_init, target_pos)

        # 3. MPPI Weighting
        min_cost = jnp.min(costs)
        weights = jax.nn.softmax(-(costs - min_cost) / self.lambda_)
        
        # 4. Update Mean
        delta = jnp.sum(weights[:, None, None] * (samples - mean_coeffs), axis=0)
        new_mean = mean_coeffs + delta
        
        new_mean = jnp.clip(new_mean, self.u_min, self.u_max)
        return new_mean, samples, weights, costs

# =========================================================
# 4. Simulation
# =========================================================

def run(save_gif=True, gif_filename="baseline_result.gif"):
    DT = 0.1
    HORIZON = 40       
    N_CP = 10           
    N_SAMPLES = 500    
    TEMP = 0.5         
    
    U_MIN = jnp.array([-1.5, -2.0])
    U_MAX = jnp.array([ 1.5,  2.0])
    
    mppi = BSplineMPPI(HORIZON, N_CP, DT, N_SAMPLES, TEMP, U_MIN, U_MAX)
    
    state = jnp.array([0.0, 0.0, 0.0, 0.0, 0.0])
    target_pos = jnp.array([10.0, 0.0])
    
    mean_coeffs = jnp.zeros((N_CP, 2))
    key = jax.random.PRNGKey(42)
    traj_hist = [state[:2]]
    
    frames = []
    energy_history = []

    print("Simulation Running...")
    
    plt.figure(figsize=(12, 5))
    
    for t in range(300):
        key, subkey = jax.random.split(key)
        
        mean_coeffs, samples, weights, costs = mppi.step(subkey, mean_coeffs, state, target_pos)
        
        free_energy = compute_free_energy(costs, mppi.lambda_)
        energy_history.append(free_energy)

        u_seq = mppi.get_control_sequence(mean_coeffs)
        u_curr = u_seq[0]
        
        state = step_dynamics(state, u_curr, DT)
        traj_hist.append(state[:2])
        
        dist_to_goal = np.linalg.norm(state[:2] - target_pos)
        
        if t % 1 == 0:
            plt.clf()
            ax = plt.gca()
            
           # Draw Obstacles
            ax.add_patch(Rectangle((3, 0.2), 6, 2.8, color='k', alpha=0.6, label='Obstacle'))
            ax.add_patch(Rectangle((3, -3.0), 6, 2.8, color='k', alpha=0.6))
            
            # Draw Safe Bounds (Dotted Line)
            plt.plot([0, 3, 3, 9, 9, 12], [3, 3, 0.2, 0.2, 3, 3], 'k--', alpha=0.3)
            plt.plot([0, 3, 3, 9, 9, 12], [-3, -3, -0.2, -0.2, -3, -3], 'k--', alpha=0.3)
            
            # Top Samples
            top_idx = jnp.argsort(weights)[-20:]
            top_samples = samples[top_idx]
            rollout_viz = jax.vmap(mppi.rollout, in_axes=(0, None))
            top_trajs = rollout_viz(top_samples, state)
            
            for k in range(len(top_samples)):
                alpha = 0.1 + 0.8 * (k / len(top_samples))
                plt.plot(top_trajs[k, :, 0], top_trajs[k, :, 1], 'g-', alpha=alpha, linewidth=1)
                
            # Optimal & History
            opt_traj = mppi.rollout(mean_coeffs, state)
            plt.plot(opt_traj[:, 0], opt_traj[:, 1], 'b-', linewidth=3, label='MPPI Plan')
            hist = np.array(traj_hist)
            plt.plot(hist[:, 0], hist[:, 1], 'r--', linewidth=2, label='Driven Path')
            plt.plot(target_pos[0], target_pos[1], 'rx', markersize=10, markeredgewidth=2)
            

            plt.title(f"Step {t} | Dist: {dist_to_goal:.2f}m | V:{state[4]:.2f}")
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
    run(save_gif=True, gif_filename="baseline_result.gif")