#optimal

#Apply dif - Big changse to apply dif prange + jit to parallelize and cause a significant speed up,
#switching from dictionaries to numpy arrays
#visualize_particle_movement similar to apply dif
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import random
from numba import jit, prange
from timeit import default_timer as timer
import cProfile


@jit
def grayscale(img_arr):
    return np.array(Image.fromarray(img_arr).convert('L'))

@jit
def find_edges(image_arr):
    return cv2.Canny(image_arr, 100, 200)

def create_particles(edges):
    particle_coords = np.argwhere(edges > 0)
    particles = np.zeros((len(particle_coords), 3), dtype=np.int64)
    particles[:, :2] = particle_coords
    particles[:, 2] = edges[tuple(particle_coords.T)]
    return particles

@jit
def apply_diffusion(particles, diffusion_time, edges_shape):
    for _ in range(diffusion_time):
        for i in prange(len(particles)):
            x, y, color = particles[i]
            dx, dy = random.choice([(0, 1), (0, -1), (1, 0), (-1, 0)])
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < edges_shape[0] and 0 <= new_y < edges_shape[1]:
                particles[i, :2] = new_x, new_y

def visualize_particle_movement(particles, edges):
    plt.figure(figsize=(10, 10))
    plt.imshow(edges, cmap='gray')
    plt.scatter(particles[:, 1], particles[:, 0], color='white', s=1)
    plt.axis('off')
    plt.show()

def runcode(path, diffusion_time):
    with Image.open(path) as img:
        image_arr = np.array(img)
    gray = grayscale(image_arr)
    edges = find_edges(gray)
    particles = create_particles(edges)
    apply_diffusion(particles, diffusion_time, edges.shape)
    visualize_particle_movement(particles, edges)

start = timer()
runcode("/content/jokic.png", diffusion_time=100)
print('optimized time: ',timer()-start)
