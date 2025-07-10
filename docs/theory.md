# Theory

## What is tortuosity

A voxel image is basically a 2D/3D array of 0s and 1s denoting solid and void, respectively (see figure below).

![Voxel image](./assets/binary.svg)

The tortuosity factor is a geometric property of the medium loosely defined as the extra length molecules need to traverse on average (via diffusion) to travel between opposing faces, e.g., $x=0$ and $x=\ell_x$, normalized by the direct length (e.g., $\ell_x$). Clearly, the tortuosity factor is direction-dependent and can be computed along each of the main principal axes of the image.

With this definition, open space has a tortuosity factor of 1, while a maze has a tortuosity factor equal to the length of the maze divided by the direct length.

## Computing tortuosity

To compute $\tau$, one needs to solve the steady state heat equation (i.e., the Laplace equation):

```math
\nabla \cdot (D_b \nabla c) = 0
```

on a voxel image with Dirichlet boundary conditions imposed on opposing faces, e.g. $c(x=0) = c_i$ and $c(x=\ell_x) = c_o$. The tortuosity factor, $\tau$, is then defined as:

```math
\tau = \frac{D_b}{D_{eff}} \varepsilon
```

where $D_{eff}$ is the effective diffusivity and can be computed using the following formula:

```math
D_{eff} = \frac{\dot{m} \cdot \ell_x}{\Delta c \cdot A}
```

where $\dot{m}$ is the mass flow rate, $\Delta c$ is the concentration difference, and $A$ is the cross-sectional area.
