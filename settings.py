class SingletonMeta(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            instance = super().__call__(*args, **kwargs)
            cls._instances[cls] = instance
        return cls._instances[cls]


class GameSettings(metaclass=SingletonMeta):
    def __init__(self):
        self.GENERATION_MODE = 'rooms_and_corridors' # 'random', 'recursive_backtracking', 'string', 'prim', 'spiral', 'growing_tree', 'rooms_and_corridors'
        self.FIELD_SIZES = (50, 50)
        self.OWN_MAP = ""
        self.CELL_SIZE = 20
        self.PLAYER_COLOR = (255, 0, 0)
        self.WALL_COLOR = (0, 0, 0)
        self.PATH_COLOR = (0, 255, 0)
        self.EMPTY_COLOR = (255, 255, 255)
        self.EXIT_COLOR = (0, 0, 255)
        self.BUTTON_COLOR = (0, 128, 255)
        self.BUTTON_HOVER_COLOR = (0, 102, 204)
        self.TEXT_COLOR = (255, 255, 255)
        self.PLAYER_MOVE_SPEED = 100  # Milliseconds per cell (higher is slower)
        self.TEST_MAP = """
        ++++.++++++++++.++++
+.................+
+.+.+.+.+.+.+.+.+.+.+
+.+.+.+.+.+.+.+.+...+
+.................+
++++.+++++.+++++++++
"""

    def __repr__(self) -> str:
        return f"GameSettings(GENERATION_MODE={self.GENERATION_MODE}, FIELD_SIZES={self.FIELD_SIZES}, OWN_MAP={len(self.OWN_MAP)} CELL_SIZE={self.CELL_SIZE})"