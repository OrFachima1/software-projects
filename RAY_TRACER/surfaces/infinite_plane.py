import numpy as np
from collections import namedtuple

Intersection = namedtuple('Intersection', ['point', 'normal', 'material_index', 't'])

class Plane:
    def __init__(self, normal, offset, material_index):
        self.normal = np.array(normal, dtype=np.float64)  # Ensure float64 type
        self.normal /= np.linalg.norm(self.normal)  # Normalize the normal vector
        self.offset = float(offset)  # Ensure float type
        self.material_index = material_index

    def intersect(self, ray_origin, ray_direction):
        # Calculate the dot product of the ray direction and plane normal
        denom = np.dot(ray_direction, self.normal)
        
        # If the denominator is close to 0, the ray is parallel to the plane
        if abs(denom) < 1e-6:
            return None
        
        # Calculate the distance along the ray to the intersection point
        t = (self.offset - np.dot(ray_origin, self.normal)) / denom
        
        # If t is negative, the intersection is behind the ray origin
        if t < 0:
            return None
        
        # Calculate the intersection point
        point = ray_origin + t * ray_direction
        
        return Intersection(point=point, normal=self.normal, material_index=self.material_index, t=t)