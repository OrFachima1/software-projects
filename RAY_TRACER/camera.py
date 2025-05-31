import numpy as np

class Camera:
    def __init__(self, position, look_at, up_vector, screen_distance, screen_width):
        if screen_distance <= 0 or screen_width <= 0:
            raise ValueError("screen_distance and screen_width must be positive")

        self.position = np.array(position, dtype=np.float32)
        self.look_at = np.array(look_at, dtype=np.float32)
        self.up_vector = np.array(up_vector, dtype=np.float32)
        self.screen_distance = float(screen_distance)
        self.screen_width = float(screen_width)
        
        # Calculate camera direction and right vector
        self.direction = self.look_at - self.position
        self.direction = self.direction / np.linalg.norm(self.direction)
        
        self.right = np.cross(self.direction, self.up_vector)
        self.right = self.right / np.linalg.norm(self.right)
        
        # Recalculate up vector to ensure it's perpendicular to direction
        self.up = np.cross(self.right, self.direction)
        
        self.ray_origins = None
        self.ray_directions = None
        self.aspect_ratio = None
        self.screen_height = None

    def generate_ray_matrices(self, image_width, image_height):
        if self.ray_origins is not None and self.ray_origins.shape[1:] == (image_height, image_width):
            return self.ray_origins, self.ray_directions

        self.aspect_ratio = image_width / image_height
        self.screen_height = self.screen_width / self.aspect_ratio
        
        # Create a grid of screen coordinates
        x = np.linspace(-self.screen_width/2, self.screen_width/2, image_width)
        y = np.linspace(self.screen_height/2, -self.screen_height/2, image_height)
        xx, yy = np.meshgrid(x, y)
        
        # Calculate the points on the screen
        screen_points = (self.position[:, np.newaxis, np.newaxis] + 
                         self.direction[:, np.newaxis, np.newaxis] * self.screen_distance +
                         self.right[:, np.newaxis, np.newaxis] * xx +
                         self.up[:, np.newaxis, np.newaxis] * yy)
        
        # Generate ray directions
        self.ray_directions = screen_points - self.position[:, np.newaxis, np.newaxis]
        self.ray_directions = self.ray_directions / np.linalg.norm(self.ray_directions, axis=0)
        
        # Create a matrix of ray origins (all the same, which is the camera position)
        self.ray_origins = np.broadcast_to(self.position[:, np.newaxis, np.newaxis], (3, image_height, image_width))
        
        return self.ray_origins, self.ray_directions

    def get_ray(self, x, y):
        if self.ray_origins is None:
            raise ValueError("Ray matrices have not been generated yet. Call generate_ray_matrices first.")
        return self.ray_origins[:, y, x], self.ray_directions[:, y, x]
    
    def get_rays_for_row(self, y, width):
        if self.ray_origins is None or self.ray_directions is None:
            raise ValueError("Ray matrices have not been generated yet. Call generate_ray_matrices first.")
        
        return self.ray_origins[:, y, :], self.ray_directions[:, y, :]