import torch
import math

def create_ray_grid(
    camera_position: torch.Tensor,
    H: int,
    W: int,
    focal: float = 1.0,
    scaling_factor: float = 1.0,
    type: str = "concat",      # "concat", "sphere", or "unit_sphere"
    coords: str = "cartesian", # "cartesian" or "polar"
    device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Build a 2D grid of per-pixel rays according to the SODA paper's 
    pose-encoding approaches.

    Args:
      camera_position: (3,) the camera center o = [ox, oy, oz].
      H, W: image height and width.
      focal: focal length for the pinhole model.
      scaling_factor: s_d for 'sphere' type; see SODA Appendix G.3.
      type: one of {"concat", "sphere", "unit_sphere"}:
          - "concat":    returns [o, d]        => shape depends on coords
          - "sphere":    returns o + s_d * d   => shape depends on coords
          - "unit_sphere": intersect each ray with the unit sphere => shape depends on coords
      coords: one of {"cartesian", "polar"}:
          - "cartesian": returns (x,y,z) or [o_x,o_y,o_z, d_x,d_y,d_z]
          - "polar":     returns (rho, azim, elev) or for unit_sphere => (azim, elev)
    
    Returns:
      A tensor of shape:
        1) if type="concat":
           - coords="cartesian": (H,W,6)
           - coords="polar":     (H,W,6)
        2) if type="sphere":
           - coords="cartesian": (H,W,3)
           - coords="polar":     (H,W,3)
        3) if type="unit_sphere" & coords="polar":
           - (H,W,2)
    """

    ################################################################
    # 1) Build per-pixel directions d in world space
    ################################################################
    # Make sure camera_position is shape (3,)
    o = camera_position.to(device).float().view(3)  # (3,)

    # Define forward axis = -o/||o|| if looking at the origin
    forward = -o / (o.norm() + 1e-9)

    # Build a naive camera coordinate frame
    world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
    right = torch.cross(forward, world_up)
    right = right / (right.norm() + 1e-9)
    new_up = torch.cross(right, forward)
    new_up = new_up / (new_up.norm() + 1e-9)

    # Rotation matrix (3,3)
    R = torch.stack([right, new_up, forward], dim=1)  # (3,3)

    # For each pixel (u,v), define normalized coords in [-1,1]
    vs, us = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing='ij'
    )
    X = (2.0 * (us + 0.5) / W - 1.0)
    Y = (1.0 - 2.0 * (vs + 0.5) / H)
    Z = torch.full_like(X, -1.0 / focal)  # assume pinhole facing -Z in camera coords

    d_cam = torch.stack([X, Y, Z], dim=-1)  # (H, W, 3)
    d_cam = d_cam / (d_cam.norm(dim=-1, keepdim=True) + 1e-9)  # normalize

    # Rotate directions into world coords
    d_cam_flat = d_cam.view(-1, 3)        # (H*W, 3)
    d_world_flat = torch.matmul(d_cam_flat, R.T)  # (H*W, 3)
    d_world = d_world_flat.view(H, W, 3)  # (H, W, 3)

    ################################################################
    # 2) Depending on 'type', form the raw data we want to encode
    ################################################################
    if type == "concat":
        # We'll replicate o across all pixels and stack [o, d]
        # shape => (H, W, 3) for o, (H, W, 3) for d => (H, W, 6).
        # Just in raw Cartesian first; if coords="polar" we convert below.
        o_broadcast = o.view(1,1,3).expand(H, W, 3)  # (H,W,3)
        raw_concat = torch.cat([o_broadcast, d_world], dim=-1)  # (H,W,6)

    elif type == "sphere":
        # r = o + s_d * d  => shape (H,W,3)
        raw_sphere = o.view(1,1,3) + scaling_factor * d_world

    elif type == "unit_sphere":
        # solve for s s.t. ||o + s d||=1, pick the smaller positive root
        # then r = o + s d => shape (H,W,3)
        # eventually we'll convert to polar and keep (azim, elev)
        o_dot_o = o.dot(o)
        d_world_flat = d_world.view(-1,3)
        o_dot_d = torch.einsum("ij,j->i", d_world_flat, o)  # (H*W,)

        b = 2.0 * o_dot_d
        c = o_dot_o - 1.0
        disc = b**2 - 4.0*c
        disc_sqrt = torch.sqrt(torch.clamp(disc, min=0.0))

        s1 = (-b - disc_sqrt) / 2.0
        s2 = (-b + disc_sqrt) / 2.0
        # pick smaller positive root
        mask1 = (s1 > 0)
        mask2 = (s2 > 0)
        s = torch.where(
            mask1 & mask2, torch.minimum(s1, s2),
            torch.where(mask1, s1, s2)
        )
        r_flat = o + s.unsqueeze(-1)*d_world_flat  # (H*W,3)
        raw_unit = r_flat.view(H, W, 3)

    else:
        raise ValueError(f"Unknown type={type}. Must be concat, sphere, or unit_sphere.")

    ################################################################
    # 3) Convert from Cartesian to coords="cartesian" or "polar"
    ################################################################

    def cart_to_polar(cart_xyz: torch.Tensor):
        """
        Convert a (H,W,3) tensor from (x,y,z) -> (rho, azim, elev).
          rho = sqrt(x^2 + y^2 + z^2)
          azim = atan2(x, z)
          elev = asin(y / rho)
        Returns (H,W,3).
        """
        eps = 1e-9
        x = cart_xyz[..., 0]
        y = cart_xyz[..., 1]
        z = cart_xyz[..., 2]

        rho = cart_xyz.norm(dim=-1) + eps
        azim = torch.atan2(x, z + eps)
        elev = torch.asin(torch.clamp(y / rho, -1.0, 1.0))
        return torch.stack([rho, azim, elev], dim=-1)  # (H,W,3)

    if type == "concat":
        if coords == "cartesian":
            # shape (H,W,6), already done.
            return raw_concat
        elif coords == "polar":
            # convert o -> polar, d -> polar, then cat => (H,W,6)
            # Step 1: split raw_concat => (o, d)
            o_cat = raw_concat[..., :3]  # (H,W,3)
            d_cat = raw_concat[..., 3:]  # (H,W,3)
            o_polar = cart_to_polar(o_cat)
            d_polar = cart_to_polar(d_cat)
            return torch.cat([o_polar, d_polar], dim=-1)  # (H,W,6)
        else:
            raise ValueError(f"coords={coords} not recognized")

    elif type == "sphere":
        if coords == "cartesian":
            # raw_sphere is (H,W,3) => done
            return raw_sphere
        elif coords == "polar":
            # convert raw_sphere to polar => (H,W,3)
            return cart_to_polar(raw_sphere)
        else:
            raise ValueError(f"coords={coords} not recognized")

    elif type == "unit_sphere":
        # we always have raw_unit in shape (H,W,3), but final depends on coords
        # You specifically requested if type="unit_sphere" & coords="polar" => (H,W,2)
        if coords == "cartesian":
            raise ValueError("type='unit_sphere' with coords='cartesian' not in your requested combos.")
        elif coords == "polar":
            # raw_unit is on the unit sphere => radius=1 => store only (azim, elev) => (H,W,2)
            # but let's confirm the request:
            # "5 - type = 'unit_sphere' & coords='polar' -> returns the grid containing 2D rays in polar coordinates"
            # We'll define them as (azim, elev).
            eps = 1e-9
            x = raw_unit[..., 0]
            y = raw_unit[..., 1]
            z = raw_unit[..., 2]

            azim = torch.atan2(x, z + eps)
            # radius is ~1, so elev = arcsin(y/rho) ~ arcsin(y)
            rho = raw_unit.norm(dim=-1) + eps  # ~1
            elev = torch.asin(torch.clamp(y / rho, -1.0, 1.0))

            return torch.stack([azim, elev], dim=-1)  # (H,W,2)
        else:
            raise ValueError(f"coords={coords} not recognized")
          
          
          
          
          
# def create_sphere_ray_grid(
#     camera_position: torch.Tensor,
#     H: int,
#     W: int,
#     focal: float = 1.0,
#     device: torch.device = torch.device("cpu"),
#     return_polar: bool = False
# ) -> torch.Tensor:
#     """
#     Cast rays from camera_position onto a unit sphere at the origin.
#     Returns either a (H, W, 3) Cartesian grid of points on the sphere (default),
#     or a (H, W, 2) grid of polar angles if return_polar=True.

#     Args:
#         camera_position: (3,) the camera center o = [ox, oy, oz]
#         H, W: image height and width
#         focal: focal length (pinhole assumption). 
#                Adjust as appropriate for your camera intrinsics.
#         return_polar: if True, convert each 3D sphere point to (azim, elev) in radians

#     Returns:
#         If return_polar=False, a float tensor (H, W, 3) of sphere points r = o + s*d
#         If return_polar=True, a float tensor (H, W, 2) of (azim, elev) angles
#     """
#     # 1) Make sure camera_position is shape (3,)
#     o = camera_position.to(device).float().view(3)  # (3,)

#     # 2) Define the camera's forward axis as -o/||o|| if looking at origin
#     #    (assuming o != (0,0,0))
#     forward = -o / (o.norm() + 1e-9)
    
#     # 3) Define a simple up vector, then compute right & actual up
#     world_up = torch.tensor([0.0, 1.0, 0.0], device=device)
#     right = torch.cross(forward, world_up)
#     right = right / (right.norm() + 1e-9)
#     new_up = torch.cross(right, forward)
#     new_up = new_up / (new_up.norm() + 1e-9)

#     # 4) Build a rotation matrix R = [right; new_up; forward], shape (3,3)
#     R = torch.stack([right, new_up, forward], dim=1)  # (3,3)

#     # 5) For each pixel (u,v), define normalized device coords in [-1,1]
#     #    We'll do a meshgrid in pixel space
#     vs, us = torch.meshgrid(
#         torch.arange(H, device=device), 
#         torch.arange(W, device=device), 
#         indexing='ij'
#     )  # each (H, W)

#     # Convert (u, v) to [-1,1], with (0,0) ~ top-left
#     # A common convention (pinhole facing -Z in camera coords):
#     X = (2.0 * (us + 0.5) / W  - 1.0)  # in [-1,1]
#     Y = (1.0 - 2.0 * (vs + 0.5) / H)   # in [-1,1]
#     Z = -1.0 / focal  # a simple negative z for pinhole

#     # 6) Construct directions in camera coords, shape (H, W, 3)
#     d_cam = torch.stack([X, Y, Z], dim=-1)  # (H, W, 3)
#     # normalize them so they are unit length in camera space
#     d_cam = d_cam / (d_cam.norm(dim=-1, keepdim=True) + 1e-9)

#     # 7) Rotate directions into WORLD space: d_world = R @ d_cam
#     #    We'll flatten then unflatten for efficiency.
#     d_cam_flat = d_cam.view(-1, 3)               # (H*W, 3)
#     d_world_flat = torch.matmul(d_cam_flat, R.T) # (H*W, 3)
#     d_world = d_world_flat.view(H, W, 3)         # (H, W, 3)
    
#     # 8) Solve for s such that |o + s*d| = 1 for each pixel
#     #    The intersection with the unit sphere yields a quadratic:
#     #        ||o + s d||^2 = 1  -->  s^2(d·d) + 2 s(o·d) + (o·o -1)=0
#     #
#     #    We'll pick the smaller positive root if it exists (the "front" intersection).
#     #
#     #    Because we normalized d_world, (d·d)=1. Let’s define:
#     #        a = 1
#     #        b = 2(o·d)
#     #        c = o·o - 1
#     #
#     #    => s = (-b ± sqrt(b^2 - 4ac)) / 2a
#     #
#     #    Then we pick whichever root is positive and smaller.
#     #
#     o_dot_o = o.dot(o)
    
#     # We'll have to do this for each pixel's direction. So let's flatten again.
#     d_world_flat = d_world.view(-1, 3)
#     o_dot_d = torch.einsum('ij,j->i', d_world_flat, o)  # (H*W,)
    
#     b = 2.0 * o_dot_d      # (H*W,)
#     c = o_dot_o - 1.0      # scalar

#     disc = b**2 - 4.0*c    # (H*W,)
#     # We'll assume disc>=0 for all pixels that actually see the sphere. 
#     # In practice, you might clamp or skip invalid ones.
#     disc_sqrt = torch.sqrt(torch.clamp(disc, min=0.0))

#     s1 = (-b - disc_sqrt) / 2.0
#     s2 = (-b + disc_sqrt) / 2.0
    
#     # pick the small positive root
#     # We'll define a small helper
#     def pick_valid_root(s1, s2):
#         # both shape (H*W,)
#         mask1 = (s1 > 0)
#         mask2 = (s2 > 0)
#         # pick s1 if s1>0 else s2, if both>0 pick min
#         # we can do this in a piecewise way:
#         valid_s = torch.where(
#             mask1 & mask2, torch.minimum(s1, s2),
#             torch.where(mask1, s1, s2)
#         )
#         return valid_s

#     s_flat = pick_valid_root(s1, s2)
#     # shape (H*W,)

#     # 9) Now compute r = o + s*d for each pixel
#     r_flat = o + s_flat.unsqueeze(-1) * d_world_flat  # (H*W, 3)
#     r = r_flat.view(H, W, 3)  # (H, W, 3)

#     # 10) If requested, convert that 3D point on the sphere to (azim,elev)
#     #     We'll define a standard spherical param:
#     #       radius = 1 (by construction)
#     #       elev   = arcsin(y)
#     #       azim   = atan2(x, z)
#     #
#     #     or any other consistent definition you prefer.
#     if return_polar:
#         x = r[..., 0]
#         y = r[..., 1]
#         z = r[..., 2]
#         eps = 1e-9
#         # elev in [-pi/2, pi/2]
#         elev = torch.asin(torch.clamp(y, -1.0+eps, 1.0-eps))
#         # azim in [-pi, pi]
#         azim = torch.atan2(x, z + eps)
#         r_polar = torch.stack([azim, elev], dim=-1)  # (H, W, 2)
#         return r_polar
#     else:
#         # Return the Cartesian 3D points on the sphere
#         return r