import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d

class Lattice:
    def __init__(self, lattice, a = None):
        if a is None:
            self.a = np.linalg.norm(lattice[0])
            self.lattice = lattice/self.a
        else:
            self.a = a
            self.lattice = lattice
        self.reciprocal_lattice = np.linalg.inv(self.lattice).T

    def __repr__(self):
        return f"Lattice({self.a*self.lattice})"

    def __str__(self):
        return f"Lattice(a: {self.a},\nlattice: {self.lattice},\nreciprocal_lattice: {self.reciprocal_lattice})"

def get_neighboring_cells(lattice, steps = 1):
    norms = np.linalg.norm(lattice, axis = 1)
    sorted_indices = np.argsort(norms)
    sorted_norms = norms[sorted_indices]

    n_max = np.array([int(np.ceil(norms[-1]/n)) for n in sorted_norms])

    coeffs = (steps+1)*n_max
    all_vectors = [[coeff*v for coeff in range(-coeffs[i], coeffs[i] + 1)] for i, v in enumerate(lattice)]
    candidates = [np.zeros_like(lattice[0])]
    for vectors in all_vectors:
        new_candidates = []
        for candidate in candidates:
            for v in vectors:
                new_candidates.append(candidate + v)
        candidates = new_candidates
    candidates = sorted(candidates, key = lambda v: np.linalg.norm(v))

    norms = {round(np.linalg.norm(candidate), ndigits = 8) for candidate in candidates}

    candidates = [[candidate for candidate in candidates if abs(np.linalg.norm(candidate) - norm) < 1e-8] for norm in sorted(norms)]

    return candidates[1:]


l1 = Lattice(np.array([[np.sqrt(3)/2, 3/2], [0, 3]]), a = 1)
l2 = Lattice(np.array([l1.lattice[0]*6, l1.lattice[1]]))

r1 = l1.reciprocal_lattice/l1.a
r2 = l2.reciprocal_lattice/l2.a

neighbours1 = get_neighboring_cells(r1, 4)
neighbours2 = get_neighboring_cells(r2, 4)

print (neighbours2)

cells1 = [np.array([0, 0])] + [cell for shell in neighbours1 for cell in shell]
cells2 = [np.array([0, 0])] + [cell for shell in neighbours2 for cell in shell]

vor1 = Voronoi(cells1)

import matplotlib.pyplot as plt
fix, ax = plt.subplots(nrows = 1, ncols = 1)
ax.scatter([cell[0] for cell in cells2], [cell[1] for cell in cells2], color = 'blue')
fig = voronoi_plot_2d(vor1, ax = ax, show_vertices = False)

ax.arrow(x = 0, y = 0, dx = r1[0, 0], dy = r1[0, 1], width = 0.005, color = 'black', length_includes_head = True)
ax.arrow(x = 0, y = 0, dx = r1[1, 0], dy = r1[1, 1], width = 0.005, color = 'black', length_includes_head = True)
ax.arrow(x = 0, y = 0, dx = r2[0, 0], dy = r2[0, 1], width = 0.002, linestyle = '--', color = 'blue', length_includes_head = True)
ax.arrow(x = 0, y = 0, dx = r2[1, 0], dy = r2[1, 1], width = 0.002, linestyle = '--', color = 'blue', length_includes_head = True)

K = r1[0]/3
Kp = -r1[0]/3

ax.scatter([K[0], Kp[0]], [K[1], Kp[1]], color = 'red')

ax.set_xlim(-0.7, 1.3)
ax.set_ylim(-0.5, 0.5)
ax.set_xticks([])
ax.set_yticks([])
plt.show()

