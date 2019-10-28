import sys
import os
sys.path.append(os.pardir)

import numpy as np
from alifebook_lib.visualizers import MatrixVisualizer

#initialize visualizer
visualizer = MatrixVisualizer()

#parameters for simulation
SPACE_GRID_SIZE = 256
dx = 0.01
dt = 1
#update span
VISUALIZATION_STEP = 8
SQUARE_SIZE = 20

#model parameters
Du = 2e-5
Dv = 1e-5
#amorphous
f, k = 0.04, 0.06
# f, k = 0.035, 0.065  # spots
# f, k = 0.012, 0.05  # wandering bubbles
# f, k = 0.025, 0.05  # waves
# f, k = 0.022, 0.051 # stripe

#initialize
u = np.ones((SPACE_GRID_SIZE, SPACE_GRID_SIZE))
v = np.zeros((SPACE_GRID_SIZE, SPACE_GRID_SIZE))

#put square of side length: SQUARE_SIZE at center
u[SPACE_GRID_SIZE // 2 - SQUARE_SIZE // 2:SPACE_GRID_SIZE // 2 + SQUARE_SIZE // 2,
SPACE_GRID_SIZE // 2 - SQUARE_SIZE // 2:SPACE_GRID_SIZE // 2 + SQUARE_SIZE // 2] = 0.5

v[SPACE_GRID_SIZE // 2 - SQUARE_SIZE // 2:SPACE_GRID_SIZE // 2 + SQUARE_SIZE // 2,
SPACE_GRID_SIZE // 2 - SQUARE_SIZE // 2:SPACE_GRID_SIZE // 2 + SQUARE_SIZE // 2] = 0.25

#noise to disrupt symmetry
u += np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE) * 0.1
v += np.random.rand(SPACE_GRID_SIZE, SPACE_GRID_SIZE) * 0.1

#visualizer returns false when window is close
while visualizer:
	for i in range(VISUALIZATION_STEP):
		#calculate laplacian
		laplacian_u = (
			np.roll(u, 1, axis=0) + np.roll(u, -1, axis=0) +
			np.roll(u, 1, axis=1) + np.roll(u, -1, axis=1) - 4 * u
		) / (dx**2)

		laplacian_v = (
			np.roll(v, 1, axis=0) + np.roll(v, -1, axis=0) + 
			np.roll(v, 1, axis=1) + np.roll(v, -1, axis=1) - 4 * v
		) / (dx**2)

		#gray-scott model equation
		du_dt = Du * laplacian_u - u * np.power(v, 2) + f * (1.0 - u)
		dv_dt = Dv * laplacian_v + u * np.power(v, 2) - (f + k) * v

		u += dt * du_dt
		v += dt * dv_dt
	visualizer.update(u)

