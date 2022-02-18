class Config:
    lgn_cnt: int
    cortex_cnt: int
    vis_angle_deg: float
    nest_thread_cnt: int
    sim_step_ms: float
    
    def __init__(self) -> None:
        # Reduced because of high complexity during connection of neurons.
        self.lgn_cnt = 20
        self.cortex_cnt = 40
        self.vis_angle_deg = 2.0
        self.nest_thread_cnt = 8
        self.sim_step_ms = 1.0

    def with_lgn_cnt(self, lgn_cnt: int) -> "Config":
        self.lgn_cnt = lgn_cnt
        return self

    def with_cortex_cnt(self, cortex_cnt: int) -> "Config":
        self.cortex_cnt = cortex_cnt
        return self
    
    def with_vis_angle_deg(self, vis_angle_deg: float) -> "Config":
        self.vis_angle_deg = vis_angle_deg
        return self
    
    def with_nest_threads(self, nest_thread_cnt: int) -> "Config":
        self.nest_thread_cnt = nest_thread_cnt
        return self
    
    def with_sim_step_ms(self, sim_step_ms: float) -> "Config":
        self.sim_step_ms = sim_step_ms
        return self
