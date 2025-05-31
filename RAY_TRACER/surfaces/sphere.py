import numpy as np
from collections import namedtuple

Intersection = namedtuple('Intersection', ['point', 'normal', 'material_index', 't'])

class Sphere:
    def __init__(self, center, radius, material_index):
        self.center = np.array(center)
        self.radius = radius
        self.radius_squared = radius * radius
        self.material_index = material_index

    def intersect(self, ray_origin, ray_direction):
        oc = ray_origin - self.center
        a = np.dot(ray_direction, ray_direction)
        b = 2 * np.dot(oc, ray_direction)
        c = np.dot(oc, oc) - self.radius_squared
        discriminant = b * b - 4 * a * c
        
        if discriminant < 0:
            return None
        
        sqrt_discriminant = np.sqrt(discriminant)
        t1 = (-b - sqrt_discriminant) / (2 * a)
        t2 = (-b + sqrt_discriminant) / (2 * a)
        
        if t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        else:
            return None

        point = ray_origin + t * ray_direction
        normal = (point - self.center) / self.radius
        
        return Intersection(point=point, normal=normal, material_index=self.material_index, t=t)