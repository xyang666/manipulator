# E-Walker-inspired 7-DoF manipulator

This is a research reconstruction, not a flight-qualified E-Walker model.

The kinematic architecture follows Nair et al., *Design engineering a walking
robotic manipulator for in-space assembly missions*, Frontiers in Robotics and
AI 9 (2022), DOI `10.3389/frobt.2022.995813`:

- symmetric seven-revolute-joint chain;
- three-axis shoulder, one-axis elbow, three-axis wrist;
- two primary tubular links;
- approximately 1.3 m total length, matching the paper's 1:6 Earth-analogue
  scale rather than the approximately 8 m full-scale concept.

The paper does not publish a URDF, joint coordinate frames, complete mass
distribution, inertia tensors, or joint limits. Those missing quantities are
explicit modelling assumptions in `model_metadata.json`. Masses and inertias
are engineering simulation values derived from simple cylinders; they must not
be presented as measured E-Walker or flight-hardware parameters.

In particular, "geometry based on the public paper" means the published
seven-joint architecture, symmetry, tubular-link form and approximately 1.3 m
prototype envelope. The exact axial allocation (0.12/0.09/0.09/0.25/0.25/
0.09/0.09/0.32 m), radii and joint axes are reconstruction choices, not
dimensions recovered from an E-Walker CAD model.

The analytical planner and MuJoCo model deliberately use matching capsule
axes and radii for collision checking. URDF collision/visual elements are
primitive cylinders with those axes and radii because URDF has no native
capsule element. There is no unpublished CAD content.

The reconstructed URDF has a modelled total mass of 10.6 kg. This is deliberately
recorded separately from the paper's approximate 12 kg prototype-level figure:
the publication does not provide the link-by-link mass distribution needed to
identify inertial parameters.
