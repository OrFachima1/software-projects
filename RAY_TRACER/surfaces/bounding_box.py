import numpy as np

class BoundingBox:
    def __init__(self, min_point, max_point):
        self.min_point = np.array(min_point)
        self.max_point = np.array(max_point)

    def center(self):
        return (self.min_point + self.max_point) / 2

    def surface_area(self):
        sides = self.max_point - self.min_point
        return 2 * (sides[0] * sides[1] + sides[1] * sides[2] + sides[2] * sides[0])

    def union(self, other):
        min_point = np.minimum(self.min_point, other.min_point)
        max_point = np.maximum(self.max_point, other.max_point)
        return BoundingBox(min_point, max_point)

    def intersect(self, ray_origin, ray_direction):
   
        inv_direction = 1.0 / (ray_direction + 1e-8)  # Add small epsilon to avoid division by zero
        t_min = (self.min_point - ray_origin) * inv_direction
        t_max = (self.max_point - ray_origin) * inv_direction
        
        t1 = np.minimum(t_min, t_max)
        t2 = np.maximum(t_min, t_max)
        
        t_near = np.max(t1)
        t_far = np.min(t2)
        
        return t_near, t_far