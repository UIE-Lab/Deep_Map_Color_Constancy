# Depth Map Color Constancy

[cite_start]This repository contains a Python implementation of the paper **"Depth map color constancy"**, published by Marc Ebner and Johannes Hansen in 2013[cite: 4, 5].

[cite_start]The main goal of this project is to use the depth map information corresponding to an RGB image to eliminate the effects of illumination and reveal the true colors of objects[cite: 8].

## Core Concept

[cite_start]The fundamental hypothesis of the algorithm is that sharp depth discontinuities in a scene (e.g., a doorway separating two different rooms) often delineate regions with different illumination conditions[cite: 62]. [cite_start]Based on this information, the algorithm prevents color averaging across boundaries where such depth jumps occur[cite: 158]. [cite_start]This allows for a more accurate illuminant estimation for distinct regions, leading to a more successful color correction[cite: 494].

## Example Result

Below is an example of the algorithm's effect on a sample scene.

![Example Result](Figure 2025-09-15 231828.png)
*On the left is the original image, and on the right is the color-corrected result.*

## Usage

To run the project, ensure that the required files `im.png` (RGB image) and `im.npy` (depth data) are in the same directory as the main script.

Required Python libraries:
* `opencv-python`
* `numpy`
* `matplotlib`

After installing the libraries, you can run the main script:
```bash
python main.py
```

## Reference

This work is based on the following paper:

> Ebner, M., & Hansen, J. (2013). Depth map color constancy. [cite_start]*Bio-Algorithms and Med-Systems*, 9(4), 167-177. [cite: 3]
