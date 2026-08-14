# GR3Mini211 assets

The MJCF and mesh files in this directory were copied from
`any2track/storage/assets/fourier_gr3mini_v211` at migration time. The upstream
repository is Apache-2.0 licensed; its license text is preserved as
`UPSTREAM_LICENSE`.

At load time `robots/gr3mini211.py` removes the upstream motor actuators and the
scene-specific `*_floor` pairs. mjlab supplies 25 explicit ideal-PD actuators and
connects collision geoms to its `terrain` plane with collision masks. The original
self-collision pairs remain in the MJCF.
