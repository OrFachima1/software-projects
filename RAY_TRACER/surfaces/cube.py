import numpy as np
from collections import namedtuple

Intersection = namedtuple('Intersection', ['point', 'normal', 'material_index', 't'])

class Cube:
    def __init__(self, center, edge_length, material_index):
        self.center = np.array(center)
        self.half_edge = edge_length / 2
        self.material_index = material_index
        self.min_bound = self.center - self.half_edge
        self.max_bound = self.center + self.half_edge

    def intersect(self, ray_origin, ray_direction):
        inv_direction = 1.0 / (ray_direction + 1e-8)
        t1 = (self.min_bound - ray_origin) * inv_direction
        t2 = (self.max_bound - ray_origin) * inv_direction

        t_min = np.max(np.minimum(t1, t2))
        t_max = np.min(np.maximum(t1, t2))

        if t_max < 0 or t_min > t_max:
            return None

        t = t_min if t_min > 0 else t_max

        point = ray_origin + t * ray_direction
        normal = np.zeros(3)
        normal[np.argmin(np.abs(point - self.center))] = np.sign(self.center - point)[np.argmin(np.abs(point - self.center))]

        return Intersection(point=point, normal=normal, material_index=self.material_index, t=t)