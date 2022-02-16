class Config:
    _lgn_cnt: int
    _cortex_cnt: int
    _vis_angle_deg: float
    _nest_thread_cnt: int
    _sim_step_ms: float
    
    def __init__(self) -> None:
        self._lgn_cnt = 40
        self._cortex_cnt = 80
        self._vis_angle_deg = 2.0
        self._nest_thread_cnt = 8
        self._sim_step_ms = 1.0

    def with_lgn_cnt(self, lgn_cnt: int) -> "Config":
        self._lgn_cnt = lgn_cnt
        return self

    def with_cortex_cnt(self, cortex_cnt: int) -> "Config":
        self._cortex_cnt = cortex_cnt
        return self
    
    def with_vis_angle_deg(self, vis_angle_deg: float) -> "Config":
        self._vis_angle_deg = vis_angle_deg
        return self
    
    def with_nest_threads(self, nest_thread_cnt: int) -> "Config":
        self._nest_thread_cnt = nest_thread_cnt
        return self
    
    def with_sim_step_ms(self, sim_step_ms: float) -> "Config":
        self._sim_step_ms = sim_step_ms
        return self
