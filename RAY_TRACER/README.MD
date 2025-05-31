# 🖥️ Ray Tracer

A basic ray tracing engine written in Python for rendering 3D scenes composed of spheres, planes, and cubes.  
The ray tracer implements shadows, reflections, transparency, and soft shadows using recursive ray casting and the Phong illumination model.

---

## ✨ Features

- Supports:
  - **Spheres**, **infinite planes**, and **axis-aligned cubes**
- **Phong shading** (ambient, diffuse, specular)
- **Recursive reflection & transparency**
- **Soft shadows** via multiple shadow rays
- Configurable **recursion depth** and **image resolution**
- Scene description using a simple `.txt` format

---

## 🖼️ Example Output

![Rendered Scene](./output/original.png)

---

## 🚀 How to Run

```bash
python ray_tracer.py scenes/scene1.txt output/render.png --width 500 --height 500
```

### Arguments:

- `scenes/scene1.txt` → input scene file  
- `output/render.png` → output image file  
- `--width`, `--height` (optional) → image resolution (default: 500×500)

---

## 🛠️ Requirements

Install required Python packages:

```bash
pip install numpy pillow
```

---

## 📁 Scene File Format

Scene files define the camera, materials, and objects using the following tags:

- `cam`: camera parameters
- `mtl`: material (diffuse/specular/reflection/transparency/shininess)
- `sph`: sphere
- `pln`: plane
- `box`: cube
- `lgt`: point light source

➡️ Each element is defined per line with its parameters (see assignment PDF for full syntax).

---

## 📚 Credits

This project was developed as part of a university Computer Graphics assignment (Spring 2024).
