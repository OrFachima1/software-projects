import argparse
from PIL import Image
import numpy as np
import multiprocessing
from functools import partial
from tqdm import tqdm
import re
from tqdm import tqdm

from camera import Camera
from light import Light
from material import Material
from scene_settings import SceneSettings
from surfaces.cube import Cube
from surfaces.infinite_plane import Plane
from surfaces.sphere import Sphere
from bvh import BVH

def parse_scene_file(file_path):
    camera = None
    scene_settings = None
    materials = []
    lights = []
    surfaces = []
    
    # Regular expression to match and remove comments
    comment_pattern = re.compile(r'#.*$')

    with open(file_path, 'r') as f:
        for line in f:
            # Remove comments and strip whitespace
            line = comment_pattern.sub('', line).strip()
            if not line:
                continue
            
            parts = line.split()
            obj_type = parts[0]
            params = [float(p) for p in parts[1:]]
            
            if obj_type == "cam":
                camera = Camera(params[:3], params[3:6], params[6:9], params[9], params[10])
            elif obj_type == "set":
                scene_settings = SceneSettings(params[:3], int(params[3]), int(params[4]))
            elif obj_type == "mtl":
                material = Material(params[:3], params[3:6], params[6:9], params[9], params[10])
                materials.append(material)
            elif obj_type == "sph":
                sphere = Sphere(params[:3], params[3], int(params[4]) - 1)  # Subtract 1 for 0-based indexing
                surfaces.append(sphere)
            elif obj_type == "pln":
                plane = Plane(params[:3], params[3], int(params[4]) - 1)  # Subtract 1 for 0-based indexing
                surfaces.append(plane)
            elif obj_type == "box":
                cube = Cube(params[:3], params[3], int(params[4]) - 1)  # Subtract 1 for 0-based indexing
                surfaces.append(cube)
            elif obj_type == "lgt":
                light = Light(params[:3], params[3:6], params[6], params[7], params[8])
                lights.append(light)
            else:
                raise ValueError(f"Unknown object type: {obj_type}")
    
    return camera, scene_settings, materials, lights, surfaces

def is_point_in_shadow(point, direction, bvh, infinite_planes, max_distance):
    epsilon = 1e-5
    shadow_origin = point + direction * epsilon

    # Check for intersections with BVH
    bvh_hit = bvh.intersect(shadow_origin, direction)
    if bvh_hit and bvh_hit.t < max_distance - epsilon:
        return True

    # Check for intersections with infinite planes
    for plane in infinite_planes:
        plane_hit = plane.intersect(shadow_origin, direction)
        if plane_hit and plane_hit.t < max_distance - epsilon:
            return True

    return False

def calculate_lighting(hit_point, normal, view_direction, material, lights, bvh, infinite_planes, scene_settings):
    color = np.zeros(3, dtype=np.float32)
    
    for light in lights:
        light_dir = light.position - hit_point
        light_distance = np.linalg.norm(light_dir)
        light_dir /= light_distance
        
        # Compute shadow intensity using optimized soft shadows
        shadow_intensity = compute_soft_shadow(hit_point, light, bvh, infinite_planes, scene_settings)
        light_intensity = light.get_light_intensity(1.0 - shadow_intensity)
        
        # Early exit if the point is completely in shadow
        if light_intensity <= 1e-6:
            continue
        
        # Diffuse reflection
        N_dot_L = np.dot(normal, light_dir)
        if N_dot_L > 0:
            diffuse = material.diffuse_color * N_dot_L
        else:
            diffuse = np.zeros(3, dtype=np.float32)
        
        # Specular reflection (using Phong model)
        reflect_dir = reflect(-light_dir, normal)
        R_dot_V = max(np.dot(reflect_dir, view_direction), 0.0)
        specular = material.specular_color * (R_dot_V ** material.phong_specularity)
        
        # Combine diffuse and specular components
        color += (diffuse + specular) * light.color * light_intensity * light.specular_intensity
    
    return np.clip(color, 0, 1)

def compute_soft_shadow(hit_point, light, bvh, infinite_planes, scene_settings):
    # Create a coordinate system around the light
    light_normal = light.position - hit_point
    light_distance = np.linalg.norm(light_normal)
    light_normal /= light_distance
    
    # Use a stable method to create an orthonormal basis
    v = np.array([1, 0, 0], dtype=np.float32) if abs(light_normal[1]) > abs(light_normal[0]) else np.array([0, 1, 0], dtype=np.float32)
    light_v_right = np.cross(v, light_normal)
    light_v_right /= np.linalg.norm(light_v_right)
    light_v_up = np.cross(light_normal, light_v_right)

    N = scene_settings.root_number_shadow_rays
    grid_ratio = light.light_radius / N
    total_rays = N * N

    # Pre-compute random offsets
    random_offsets = np.random.rand(N, N, 2).astype(np.float32) - 0.5

    shadow_rays_count = 0
    for i in range(N):
        for j in range(N):
            x = grid_ratio * (i - N / 2 + 0.5 + random_offsets[i, j, 0])
            y = grid_ratio * (j - N / 2 + 0.5 + random_offsets[i, j, 1])
            
            point_on_grid = light.position + x * light_v_right + y * light_v_up
            
            grid_ray = point_on_grid - hit_point
            grid_ray_length = np.linalg.norm(grid_ray)
            grid_ray /= grid_ray_length

            if is_point_in_shadow(hit_point, grid_ray, bvh, infinite_planes, grid_ray_length):
                shadow_rays_count += 1

    return shadow_rays_count / total_rays

def cast_ray(ray_origin, ray_direction, bvh, infinite_planes, materials, lights, scene_settings, depth=0):
    if depth > scene_settings.max_recursions:
        return np.array(scene_settings.background_color, dtype=np.float32)

    # Check intersection with BVH (finite objects) and infinite planes
    bvh_hit = bvh.intersect(ray_origin, ray_direction)
    plane_hit = min((plane.intersect(ray_origin, ray_direction) for plane in infinite_planes), 
                    key=lambda x: x.t if x else float('inf'))
    
    closest_hit = min(filter(None, [bvh_hit, plane_hit]), key=lambda x: x.t, default=None)
    
    if closest_hit is None:
        return np.array(scene_settings.background_color, dtype=np.float32)

    material = materials[closest_hit.material_index]
    hit_point = ray_origin + closest_hit.t * ray_direction
    normal = closest_hit.normal
    view_direction = -ray_direction

    # Calculate direct lighting with soft shadows
    color = calculate_lighting(hit_point, normal, view_direction, material, lights, bvh, infinite_planes, scene_settings)

    # Calculate reflection
    reflection_color = np.zeros(3, dtype=np.float32)
    if np.any(material.reflection_color > 0) and depth < scene_settings.max_recursions:
        reflection_direction = reflect(ray_direction, normal)
        reflection_origin = hit_point + 1e-5 * reflection_direction
        reflection_color = cast_ray(reflection_origin, reflection_direction, bvh, infinite_planes, materials, lights, scene_settings, depth + 1)
        reflection_color *= material.reflection_color

    # Calculate refraction
    refraction_color = np.zeros(3, dtype=np.float32)
    if material.transparency > 0 and depth < scene_settings.max_recursions:
        refractive_index = 1.5  # This could be a material property
        refraction_direction = refract(ray_direction, normal, 1.0, refractive_index)
        if refraction_direction is not None:
            refraction_origin = hit_point + 1e-5 * refraction_direction
            refraction_color = cast_ray(refraction_origin, refraction_direction, bvh, infinite_planes, materials, lights, scene_settings, depth + 1)
            refraction_color *= material.transparency

    # Combine colors based on material properties
    final_color = color * (1 - material.transparency) + reflection_color + refraction_color
    
    return np.clip(final_color, 0, 1)


def reflect(incident, normal):
    return incident - 2 * np.dot(incident, normal) * normal

def refract(incident, normal, n1, n2):
    cos_i = np.clip(np.dot(incident, normal), -1, 1)
    eta = n1 / n2
    sin2_t = eta**2 * (1 - cos_i**2)
    if sin2_t > 1:
        return None  # Total internal reflection
    cos_t = np.sqrt(1 - sin2_t)
    return eta * incident - (eta * cos_i + cos_t) * normal

def process_row(args):
    y, width, camera, bvh, infinite_planes, materials, lights, scene_settings = args
    row_colors = np.zeros((width, 3), dtype=np.float32)
    
    ray_origins, ray_directions = camera.get_rays_for_row(y, width)
    for x in range(width):
        print(f'Processing {x}', end='\r')


        ray_origin, ray_direction = ray_origins[:, x], ray_directions[:, x]
        color = cast_ray(ray_origin, ray_direction, bvh, infinite_planes, materials, lights, scene_settings)
        row_colors[x] = color
    
    return row_colors



def save_image(image_array, output_path):
    # Flip the image horizontally before saving
    image_array = np.flip(image_array, axis=1)
    image = Image.fromarray(np.uint8(image_array * 255))
    image.save(output_path)


def main():
    parser = argparse.ArgumentParser(description='Python Ray Tracer')
    parser.add_argument('scene_file', type=str, nargs='?', default='scenes/original.txt', help='Path to the scene file')
    parser.add_argument('output_image', type=str, nargs='?', default='output/original.png', help='Name of the output image file')
    parser.add_argument('--width', type=int, default=500, help='Image width')
    parser.add_argument('--height', type=int, default=500, help='Image height')
    args = parser.parse_args()

    # Parse the scene file
    camera, scene_settings, materials, lights, surfaces = parse_scene_file(args.scene_file)

    # Separate infinite planes from other surfaces
    infinite_planes = [s for s in surfaces if isinstance(s, Plane)]
    finite_surfaces = [s for s in surfaces if not isinstance(s, Plane)]

    # Build BVH for finite surfaces
    bvh = BVH(finite_surfaces)

    # Generate rays for each pixel
    camera.generate_ray_matrices(args.width, args.height)

    # Create a pool of worker processes
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)

    # Prepare the arguments for multiprocessing
    process_args = [(y, args.width, camera, bvh, infinite_planes, materials, lights, scene_settings) for y in range(args.height)]

    # Process rows in parallel with progress bar
    with tqdm(total=args.height, desc="Rendering", unit="row") as pbar:
        image_array = np.array(list(tqdm(
            pool.imap(process_row, process_args),
            total=args.height,
            desc="Rendering",
            unit="row",
            leave=False
        )))
        pbar.update(args.height)

    # Close the pool
    pool.close()
    pool.join()

    # Save the output image
    save_image(image_array, args.output_image)

    print(f"Ray tracing completed. Output saved to {args.output_image}")

if __name__ == '__main__':
    main()