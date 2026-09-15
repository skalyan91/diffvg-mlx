import mlx.core as mx
import pydiffvg

class PixelFilter:
    def __init__(self,
                 type,
                 radius = mx.array(0.5)):
        self.type = type
        self.radius = radius
