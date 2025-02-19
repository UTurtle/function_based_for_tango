# noise_pipeline/constants.py



# Shape 및 Pattern의 기본 리스트
DEFAULT_SHAPES = [
    "circle", "rectangle", "ellipse", "horizontal_spike", "vertical_spike",
    "fog", "pillar", "horizontal_line", "vertical_line",
    "horizontal_range_dist_db", "vertical_range_dist_db",
    "trapezoid", "hill", "wave_pattern", "polygon"
]

DEFAULT_PATTERNS = [
    "linear", "random", "n_linear_repeat_t_sleep", "convex"
]

# Shape 생성 시 기본으로 사용하는 ratio (나오는 비중)
RATIO_SHAPE_BASE = {
    "circle": 1,
    "rectangle": 1,
    "ellipse": 0,
    "horizontal_spike": 1,
    "vertical_spike": 1,
    "fog": 0,
    "pillar": 0,
    "horizontal_line": 1,
    "vertical_line": 1,
    "horizontal_range_dist_db": 1,
    "vertical_range_dist_db": 1,
    "trapezoid": 1,
    "hill": 1,
    "wave_pattern": 1,
    "polygon": 1
}

RATIO_PATTERN_BASE = {
    "linear": 1,
    "random": 0,
    "n_linear_repeat_t_sleep": 1,
    "convex": 0
}

# 클래스 이름과 JSON에 저장할 소문자 타입 이름 간의 매핑 테이블
SHAPE_TYPE_MAPPING = {
    "CircleDBShape": "circle",
    "TrapezoidDBShape": "trapezoid",
    "RectangleDBShape": "rectangle",
    "EllipseDBShape": "ellipse",
    "FogDBShape": "fog",
    "PolygonDBShape": "polygon",
    "WavePatternDBShape": "wave_pattern",
    "RealWorldNoiseDBShape": "real_world_noise",
    "HorizontalSpikeDBShape": "horizontal_spike",
    "VerticalSpikeDBShape": "vertical_spike",
    "HillDBShape": "hill",
    "PillarDBShape": "pillar",
    "HorizontalLineDBShape": "horizontal_line",
    "VerticalLineDBShape": "vertical_line",
    "HorizontalRangeDistributionDBShape": "horizontal_range_dist_db",
    "VerticalRangeDistributionDBShape": "vertical_range_dist_db",
    # 필요한 경우 여기서 계속 확장
}

PATTERN_TYPE_MAPPING = {
    "LinearPattern": "linear",
    "RandomPattern": "random",
    "NLinearRepeatTSleepPattern": "n_linear_repeat_t_sleep",
    "ConvexPattern": "convex",
    "FunctionPattern": "function"
}
