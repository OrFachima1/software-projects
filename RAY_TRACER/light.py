import numpy as np
from scipy.ndimage import gaussian_filter

class Light:
    def __init__(self, position, color, specular_intensity, shadow_intensity, light_radius):
        self.position = np.array(position, dtype=np.float64)
        self.color = np.array(color, dtype=np.float64)
        self.specular_intensity = float(specular_intensity)
        self.shadow_intensity = float(shadow_intensity)
        self.light_radius = float(light_radius)

    def get_direction(self, hit_point):
        direction = self.position - hit_point
        return direction / np.linalg.norm(direction)
  
    def get_light_intensity(self, visibility):
        return (1 - self.shadow_intensity) + self.shadow_intensity * visibility
    
   