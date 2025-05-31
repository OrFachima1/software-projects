import numpy as np
from surfaces.bounding_box import BoundingBox
from collections import namedtuple
from surfaces.sphere import Sphere
from surfaces.cube import Cube

Intersection = namedtuple('Intersection', ['point', 'normal', 'material_index', 't'])

class BVHNode:
    def __init__(self, objects, axis=0, max_objects=4, depth=0):
        self.left = None
        self.right = None
        self.objects = objects
        self.bbox = self.compute_bounding_box(objects)

        if len(objects) <= max_objects or depth > 10:
            return

        # Compute the centroid of all objects
        centroids = np.array([obj.bbox.center() for obj in objects])
        
        # Choose the axis with the largest spread of centroids
        axis = np.argmax(np.ptp(centroids, axis=0))
        
        # Sort objects based on their centroid along the chosen axis
        sorted_objects = sorted(objects, key=lambda obj: obj.bbox.center()[axis])
        mid = len(sorted_objects) // 2

        self.left = BVHNode(sorted_objects[:mid], (axis + 1) % 3, max_objects, depth + 1)
        self.right = BVHNode(sorted_objects[mid:], (axis + 1) % 3, max_objects, depth + 1)
        self.objects = []

    def compute_bounding_box(self, objects):
        if not objects:
            return BoundingBox(np.array([0, 0, 0]), np.array([0, 0, 0]))
        
        bboxes = [obj.bbox for obj in objects]
        min_point = np.min([bbox.min_point for bbox in bboxes], axis=0)
        max_point = np.max([bbox.max_point for bbox in bboxes], axis=0)
        return BoundingBox(min_point, max_point)

    def intersect(self, ray_origin, ray_direction):
        t_min, t_max = self.bbox.intersect(ray_origin, ray_direction)
        if t_max < 0 or t_min > t_max:
            return None

        if self.objects:
            return self.intersect_objects(ray_origin, ray_direction)

        left_hit = self.left.intersect(ray_origin, ray_direction) if self.left else None
        right_hit = self.right.intersect(ray_origin, ray_direction) if self.right else None

        if left_hit and right_hit:
            return left_hit if left_hit.t < right_hit.t else right_hit
        return left_hit or right_hit

    def intersect_objects(self, ray_origin, ray_direction):
        closest_hit = None
        min_distance = float('inf')

        for obj in self.objects:
            hit = obj.intersect(ray_origin, ray_direction)
            if hit and hit.t < min_distance:
                closest_hit = hit
                min_distance = hit.t

        return closest_hit

class BVH:
    def __init__(self, objects):
        # Precompute bounding boxes for all objects
        for obj in objects:
            if not hasattr(obj, 'bbox'):
                if isinstance(obj, Sphere):
                    obj.bbox = BoundingBox(obj.center - obj.radius, obj.center + obj.radius)
                elif isinstance(obj, Cube):
                    obj.bbox = BoundingBox(obj.center - obj.half_edge, obj.center + obj.half_edge)
                # Add other object types as needed

        self.root = BVHNode(objects)

    def intersect(self, ray_origin, ray_direction):
        return self.root.intersect(ray_origin, ray_direction)