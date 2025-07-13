# 🧠 Image Abstraction via Particle Diffusion

A high-performance image processing project that abstracts an image into a dotted outline, then simulates the diffusion of those particles over time. Originally developed for **COMP-360: Introduction to High Performance Computing** at Wesleyan University, this script demonstrates significant optimization gains by leveraging parallelism and JIT compilation.

 **Completed**: Spring 2024  
 **Course**: COMP-360  
 **Input Format**: JPEG/PNG images  
 **Runtime Speedup**: ~7 minutes → **<30 seconds** for JPEGs

---

## 🌟 Highlights

- 🖼️ **Edge-Based Particle Simulation** – Converts image edges into "particles" that disperse over time
- ⚡ **Numba JIT + `prange` Optimization** – Parallelized loops for massive speedup using CPU
- 🔍 **Visual Output** – Live rendering of particle diffusion using `matplotlib`
- 🧪 **High Performance Computing Ready** – Designed to run efficiently on Wesleyan's HPC cluster
- 💾 **Efficient Memory Use** – Switched from dictionaries to NumPy arrays for minimal overhead

---

## 📖 Overview

This project loads an image, extracts its edges, and transforms those edges into particles that spread outward over a series of time steps, producing a dynamic abstract visualization of the image’s structure. It was designed to explore computational optimization techniques and benchmark improvements in execution time.

### Core Workflow

1. **Grayscale Conversion** – Input image converted to grayscale
2. **Edge Detection** – Edges identified using OpenCV's Canny filter
3. **Particle Initialization** – Each edge pixel becomes a "particle" stored in a NumPy array
4. **Diffusion Algorithm** – Simulated random walk diffusion using Numba’s parallel `prange`
5. **Visualization** – Final particle positions plotted over the original edges

---

## 🧪 Optimization Details

| Optimization Applied         | Impact                                      |
|-----------------------------|---------------------------------------------|
|  Numba JIT (`@jit`)        | Removes Python interpreter overhead         |
|  Numba Parallel Loop (`prange`) | Multithreaded execution for particle updates |
|  NumPy Arrays              | Replaced dictionaries for vectorized memory access |
|  Profiling (`timeit`, `cProfile`) | Measured impact of each optimization phase  |

These changes resulted in **runtime improvements from ~7 minutes to <30 seconds** for typical images like JPEGs, enabling feasible deployment on shared compute clusters.

---

## 🧰 Technologies Used

- `Python 3`
- `NumPy`
- `Matplotlib`
- `PIL` (Pillow)
- `OpenCV` (for edge detection)
- `Numba` (for JIT + parallelization)
- `timeit` and `cProfile` (for performance benchmarking)

---

## 🚀 How to Run

```bash
python optimized_diffusion.py
