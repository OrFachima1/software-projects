import numpy as np

class Material:
    def __init__(self, diffuse_color, specular_color, reflection_color, 
                 phong_specularity, transparency):
        self.diffuse_color = np.array(diffuse_color)
        self.specular_color = np.array(specular_color)
        self.reflection_color = np.array(reflection_color)
        self.phong_specularity = phong_specularity
        self.transparency = transparency

    def compute_diffuse(self, light_intensity, normal, light_direction):
        """Compute the diffuse component of the material"""
        return self.diffuse_color * light_intensity * max(np.dot(normal, light_direction), 0)

    def compute_specular_blinn_phong(self, light_color, normal, half_vector):
        return self.specular_color * light_color * np.power(np.maximum(np.dot(normal, half_vector), 0), self.phong_specularity)
    
    def compute_specular(self, light_intensity, normal, light_direction, view_direction):
        """Compute the specular component of the material"""
        reflection = 2 * np.dot(normal, light_direction) * normal - light_direction
        return self.specular_color * light_intensity * max(np.dot(reflection, view_direction), 0) ** self.phong_specularity

    def is_reflective(self):
        """Check if the material is reflective"""
        return np.any(self.reflection_color > 0)

    def is_transparent(self):
        """Check if the material is transparent"""
        return self.transparency > 0