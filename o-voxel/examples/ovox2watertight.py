import torch
import o_voxel
import trimesh

RES = 512

# Load data
coords, data = o_voxel.io.read("ovoxel_helmet.vxz")
dual_vertices = data['dual_vertices']
intersected = data['intersected']

# Depack
dual_vertices = dual_vertices / 255
intersected = torch.cat([
    intersected % 2,
    intersected // 2 % 2,
    intersected // 4 % 2,
], dim=-1).bool()

# Extract a watertight, consistently oriented outer surface.
# Unlike flexible_dual_grid_to_mesh (which emits one quad per intersected
# edge and therefore reproduces double layers, inner shells and open sheets),
# this recovers inside/outside signs by flood-filling from outside the object
# and only keeps the surface that bounds the enclosed volume:
#   - duplicated parallel sheets collapse to the outermost one
#   - enclosed inner geometry (mouth bags, inner clothing layers) is culled
#   - pinholes are sealed as long as the active-voxel shell is closed
# Runs on CPU (NumPy/SciPy), no CUDA extension required.
rec_verts, rec_faces = o_voxel.watertight.flexible_dual_grid_to_watertight_mesh(
    coords,
    dual_vertices,
    intersected,
    grid_size=RES,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
    verbose=True,
)

mesh = trimesh.Trimesh(
    vertices=rec_verts.cpu(), faces=rec_faces.cpu(),
    process=False
)
print(f"watertight: {mesh.is_watertight}, winding consistent: {mesh.is_winding_consistent}, "
      f"volume: {mesh.volume:.6f}")
mesh.export("rec_helmet_watertight.ply")
