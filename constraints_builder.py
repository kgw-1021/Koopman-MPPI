import jax.numpy as jnp
import jax

class ConstraintStack:
    def __init__(self):
        self.A_blocks = []
        self.proj_funcs = []
        self.dims = []

    def add_box(self, A_block, l, u):
        """ Box Constraint: l <= Ax <= u """
        self.A_blocks.append(A_block)
        m = A_block.shape[0]
        self.dims.append(m)
        
        # JAX의 closure 기능을 이용해 l, u를 캡처
        def proj(z_segment):
            return jnp.clip(z_segment, l, u)
        self.proj_funcs.append(proj)

    def add_equality(self, A_block, b):
        """ Equality Constraint: Ax = b """
        self.A_blocks.append(A_block)
        m = A_block.shape[0]
        self.dims.append(m)
        
        def proj(z_segment):
            return b # 등식 제약은 z가 항상 b여야 함 (Soft ADMM 기준)
        self.proj_funcs.append(proj)

    def add_ball(self, A_block, center, radius):
        """ Ball Constraint: ||Ax - center||_2 <= radius """
        self.A_blocks.append(A_block)
        m = A_block.shape[0]
        self.dims.append(m)
        
        def proj(z_segment):
            diff = z_segment - center
            norm = jnp.linalg.norm(diff)
            # 반지름 안이면 그대로, 밖이면 표면으로 투영
            scale = jnp.where(norm > radius, radius / (norm + 1e-6), 1.0)
            return center + diff * scale
        self.proj_funcs.append(proj)

    def build(self):
        """ Solver에 넘길 전체 A 행렬과 통합 proj_fn 생성 """
        full_A = jnp.vstack(self.A_blocks)
        
        # 블록별로 쪼개서 투영 함수 적용 후 다시 합침
        # 주의: JAX JIT 내부에서 사용되므로 리스트 인덱싱 대신 split 사용
        indices = jnp.cumsum(jnp.array(self.dims))[:-1]
        
        def composite_proj(z_full):
            # z 벡터를 블록별로 자름
            z_splits = jnp.split(z_full, indices)
            z_projected = []
            for func, z_seg in zip(self.proj_funcs, z_splits):
                z_projected.append(func(z_seg))
            return jnp.concatenate(z_projected)
            
        return full_A, composite_proj