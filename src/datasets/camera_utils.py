import torch
import numpy as np

def compute_ray_grids(
    camera_npz_path, 
    H, 
    W, 
    use_canonical=False,
    device='cpu'
):
    """
    Loads a camera.npz for a single object, and returns a (24, H, W, 6)
    ray grid for all views. Each element is [o_x, o_y, o_z, d_x, d_y, d_z].
    
    Args:
        camera_npz_path (str): Path to the 'camera.npz' file for one object.
        H (int): Image height.
        W (int): Image width.
        use_canonical (bool): If True, returns rays in canonical coordinates
            relative to view 0. If False, returns in the original world coordinates.
        device (str): PyTorch device, e.g. 'cpu' or 'cuda'.
    
    Returns:
        torch.Tensor of shape (24, H, W, 6) containing ray origins and directions 
        for each pixel and each of the 24 views.
    """
    # --------------------------------------------------------------------------
    # 1) Load camera parameters from NPZ into torch tensors
    # --------------------------------------------------------------------------
    cameras_np = np.load(camera_npz_path)
    cameras = {}
    for k in cameras_np.files:
        cameras[k] = torch.tensor(cameras_np[k], dtype=torch.float32, device=device)

    # --------------------------------------------------------------------------
    # 2) Define a rotation matrix that is applied in the SRT code 
    #    (z=0 becomes the ground plane, etc.)
    # --------------------------------------------------------------------------
    rot_mat_np = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ], dtype=np.float32)
    # rot_mat_np = np.array([
    #     [1, 0,  0, 0],
    #     [0, 1,  0, 0],
    #     [0, 0,  1, 0],
    #     [0, 0,  0, 1]
    # ], dtype=np.float32)
    rot_mat = torch.tensor(rot_mat_np, dtype=torch.float32, device=device)

    # --------------------------------------------------------------------------
    # 3) Apply the rotation to each view's world_mat_inv and world_mat
    # --------------------------------------------------------------------------
    for i in range(24):
        # world_mat_inv_i = rot_mat @ world_mat_inv_i
        cameras[f'world_mat_inv_{i}'] = rot_mat @ cameras[f'world_mat_inv_{i}']
        # world_mat_i = world_mat_i @ rot_mat^T  (rot_mat is symmetric, so T == itself here)
        cameras[f'world_mat_{i}'] = cameras[f'world_mat_{i}'] @ rot_mat.transpose(0, 1)

    # --------------------------------------------------------------------------
    # 4) Prepare a meshgrid of normalized coordinates: shape (H, W)
    #    Note: PyTorch's meshgrid has (row, col) ordering, so we do y first, then x.
    # --------------------------------------------------------------------------
    y_lin = torch.linspace(-1.0, 1.0, H, device=device)
    x_lin = torch.linspace(-1.0, 1.0, W, device=device)
    ymap, xmap = torch.meshgrid(y_lin, x_lin, indexing='ij')  # => (H, W) each

    # We'll create a (H, W, 3) volume = [x, y, 1].
    ones_map = torch.ones_like(xmap)
    # shape: (H, W, 3)
    pixel_coords = torch.stack([xmap, ymap, ones_map], dim=-1)

    # --------------------------------------------------------------------------
    # 5) Define helper functions for transforming points/directions in Torch
    # --------------------------------------------------------------------------
    def transform_points(pts, mat):
        """
        pts: (N, 3) or (H, W, 3)
        mat: (4, 4)
        returns: same shape (N, 3)/(H, W, 3), applying full transform including translation
        """
        orig_shape = pts.shape
        pts_flat = pts.reshape(-1, 3)            # (N, 3)
        ones = torch.ones((pts_flat.shape[0], 1), device=device)
        pts_hom = torch.cat([pts_flat, ones], dim=-1)  # (N, 4)

        pts_transformed = pts_hom @ mat.transpose(0, 1)  # (N, 4)
        # drop the homogeneous dimension
        pts_transformed = pts_transformed[:, :3]
        return pts_transformed.reshape(orig_shape)

    def transform_directions(dirs, mat):
        """
        dirs: (N, 3) or (H, W, 3)
        mat: (4, 4)
        returns: same shape, ignoring translation, only applying mat[:3,:3]
        """
        orig_shape = dirs.shape
        dirs_flat = dirs.reshape(-1, 3)  # (N, 3)

        # Only rotation part (top-left 3x3)
        rot = mat[:3, :3]
        dirs_rot = dirs_flat @ rot.transpose(0, 1)  # (N, 3)

        return dirs_rot.reshape(orig_shape)

    # --------------------------------------------------------------------------
    # 6) Compute ray directions and camera origins for each of the 24 views
    # --------------------------------------------------------------------------
    all_rays_6d = torch.zeros((24, H, W, 6), dtype=torch.float32, device=device)

    # If use_canonical: we'll transform everything into the canonical coords
    # of view 0, i.e. world_mat_0
    if use_canonical:
        canonical_extrinsic = cameras['world_mat_0']  # shape (4,4)

    for i in range(24):
        # (a) Compute directions by applying the combined inverse transformations 
        #     (world_mat_inv_i @ camera_mat_inv_i) to [x, y, 1].
        inv_mat = cameras[f'world_mat_inv_{i}'] @ cameras[f'camera_mat_inv_{i}']
        # shape (H, W, 3)
        dirs_world = transform_directions(pixel_coords, inv_mat)
        # normalize
        dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)

        # (b) Camera origin: last column of world_mat_inv_i
        #    In a 4x4 matrix M, the camera center in world coords is M[:3, -1].
        cam_pos_world = cameras[f'world_mat_inv_{i}'][:3, 3]  # shape (3,)

        # (c) If we're using canonical coords, transform both the origin and direction 
        if use_canonical:
            # transform the camera position with the canonical extrinsic
            cam_pos_canonical = transform_points(cam_pos_world.unsqueeze(0), canonical_extrinsic)[0]
            # transform directions with rotation only 
            dirs_canonical = transform_directions(dirs_world, canonical_extrinsic)
            
            # Now the final origin/direction 
            cam_pos_final = cam_pos_canonical
            dirs_final = dirs_canonical
        else:
            cam_pos_final = cam_pos_world
            dirs_final = dirs_world

        # (d) Expand camera pos to (H, W, 3), then concat with directions to form (H, W, 6)
        # shape (H, W, 3)
        origin_grid = cam_pos_final.view(1, 1, 3).expand(H, W, 3)
        rays_6d = torch.cat([origin_grid, dirs_final], dim=-1)  # (H, W, 6)

        # (e) Save to the final tensor
        all_rays_6d[i] = rays_6d

    return all_rays_6d


def compute_ray_grids_for_views(
    camera_npz_path, 
    H, 
    W, 
    views_idx,
    use_canonical=False,
    device='cpu'
):
    """
    Loads a camera.npz for a single object, and returns a (len(views_idx), H, W, 6)
    ray grid for the specified views. Each element is [o_x, o_y, o_z, d_x, d_y, d_z].
    
    Args:
        camera_npz_path (str): Path to the 'camera.npz' file for one object.
        H (int): Image height.
        W (int): Image width.
        views_idx (List[int]): A list of view indices (0-23).
        use_canonical (bool): If True, returns rays in canonical coordinates
            relative to view 0. If False, returns in the original world coordinates.
        device (str): PyTorch device, e.g. 'cpu' or 'cuda'.
    
    Returns:
        torch.Tensor of shape (len(views_idx), H, W, 6)
        containing ray origins and directions for each pixel and each specified view.
    """
    # --------------------------------------------------------------------------
    # 1) Load camera parameters from NPZ into torch tensors
    # --------------------------------------------------------------------------
    cameras_np = np.load(camera_npz_path)
    cameras = {}
    for k in cameras_np.files:
        cameras[k] = torch.tensor(cameras_np[k], dtype=torch.float32, device=device)

    # --------------------------------------------------------------------------
    # 2) SRT-style rotation matrix
    # --------------------------------------------------------------------------
    rot_mat_np = np.array([
        [1, 0,  0, 0],
        [0, 0, -1, 0],
        [0, 1,  0, 0],
        [0, 0,  0, 1]
    ], dtype=np.float32)
    # rot_mat_np = np.array([
    #     [1, 0,  0, 0],
    #     [0, 1,  0, 0],
    #     [0, 0,  1, 0],
    #     [0, 0,  0, 1]
    # ], dtype=np.float32)
    rot_mat = torch.tensor(rot_mat_np, dtype=torch.float32, device=device)

    # --------------------------------------------------------------------------
    # 3) Apply the rotation to world_mat_inv_i and world_mat_i
    # --------------------------------------------------------------------------
    for i in range(24):
        cameras[f'world_mat_inv_{i}'] = rot_mat @ cameras[f'world_mat_inv_{i}']
        cameras[f'world_mat_{i}'] = cameras[f'world_mat_{i}'] @ rot_mat.transpose(0, 1)

    # --------------------------------------------------------------------------
    # 4) Prepare a meshgrid of normalized coordinates: shape (H, W)
    #    We'll form (H, W, 3) => [x, y, 1]
    # --------------------------------------------------------------------------
    y_lin = torch.linspace(-1.0, 1.0, H, device=device)
    x_lin = torch.linspace(-1.0, 1.0, W, device=device)
    ymap, xmap = torch.meshgrid(y_lin, x_lin, indexing='ij')  # => (H, W)
    ones_map = torch.ones_like(xmap)
    # shape: (H, W, 3)
    pixel_coords = torch.stack([xmap, ymap, ones_map], dim=-1)

    # --------------------------------------------------------------------------
    # 5) Helpers for transforms
    # --------------------------------------------------------------------------
    def transform_points(pts, mat):
        """
        pts: (N, 3) or (H, W, 3)
        mat: (4, 4)
        returns: same shape, applying full transform including translation
        """
        orig_shape = pts.shape
        pts_flat = pts.reshape(-1, 3)   # (N, 3)
        ones = torch.ones((pts_flat.shape[0], 1), device=device)
        pts_hom = torch.cat([pts_flat, ones], dim=-1)  # (N, 4)
        
        pts_transformed = pts_hom @ mat.transpose(0, 1)  # (N, 4)
        return pts_transformed[:, :3].reshape(orig_shape)

    def transform_directions(dirs, mat):
        """
        dirs: (N, 3) or (H, W, 3)
        mat: (4, 4)
        returns: same shape, ignoring translation, only applying rotation
        """
        orig_shape = dirs.shape
        dirs_flat = dirs.reshape(-1, 3)  # (N, 3)
        rot = mat[:3, :3]               # top-left 3x3
        dirs_rot = dirs_flat @ rot.transpose(0, 1)  # (N, 3)
        return dirs_rot.reshape(orig_shape)

    # --------------------------------------------------------------------------
    # 6) Prepare canonical_extrinsic if use_canonical
    # --------------------------------------------------------------------------
    if use_canonical:
        canonical_extrinsic = cameras['world_mat_0']  # shape (4,4)

    # --------------------------------------------------------------------------
    # 7) Loop over the specified views
    # --------------------------------------------------------------------------
    num_views = len(views_idx)
    all_rays_6d = torch.zeros((num_views, H, W, 6), dtype=torch.float32, device=device)

    for idx, v in enumerate(views_idx):
        # (a) Combined inverse transform => inv_mat
        inv_mat = cameras[f'world_mat_inv_{v}'] @ cameras[f'camera_mat_inv_{v}']

        # Directions in world coordinates
        dirs_world = transform_directions(pixel_coords, inv_mat)
        dirs_world = dirs_world / torch.norm(dirs_world, dim=-1, keepdim=True)

        # Camera origin in world coords = M[:3,3]
        cam_pos_world = cameras[f'world_mat_inv_{v}'][:3, 3]

        if use_canonical:
            # Transform camera position
            cam_pos_canonical = transform_points(cam_pos_world.unsqueeze(0), canonical_extrinsic)[0]
            # Transform directions
            dirs_canonical = transform_directions(dirs_world, canonical_extrinsic)

            # Final origin/directions
            cam_pos_final = cam_pos_canonical
            dirs_final = dirs_canonical
        else:
            cam_pos_final = cam_pos_world
            dirs_final = dirs_world

        # Expand camera pos to (H, W, 3), concat with directions => (H, W, 6)
        origin_grid = cam_pos_final.view(1, 1, 3).expand(H, W, 3)
        rays_6d = torch.cat([origin_grid, dirs_final], dim=-1)  # shape: (H, W, 6)

        # Store
        all_rays_6d[idx] = rays_6d

    return all_rays_6d


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
    right = torch.cross(forward, world_up, dim=0)
    right = right / (right.norm() + 1e-9)
    new_up = torch.cross(right, forward, dim=0)
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
          
          
          
def check_ray_grid_with_pointcloud(
    camera_npz_path,
    pointcloud_npz_path,
    ray_grid,         # shape (24, H, W, 6)
    view_idx,         # which view in [0..23] you want to check
    H, W,
    device='cpu'
):
    """
    Example consistency check:
    1) Loads the point cloud
    2) Transforms points to the camera coordinate system
    3) Projects them to the [-1,1]^2 image plane
    4) Compares with the per-pixel directions from ray_grid
    
    Args:
      camera_npz_path:  path to cameras.npz for this object
      pointcloud_npz_path: path to pointcloud.npz
      ray_grid: (24, H, W, 6) -> your computed [o, d] for each pixel, for each view
      view_idx: which of the 24 views to check
      H, W: resolution
      device: 'cpu' or 'cuda'
    """
    # ----------------------- Load cameras ------------------------
    cam_np = np.load(camera_npz_path)
    cameras = {}
    for k in cam_np.files:
        cameras[k] = torch.tensor(cam_np[k], dtype=torch.float32, device=device)

    # Combined transform:  (world_mat_inv_{view_idx} @ camera_mat_inv_{view_idx})
    # or the forward transform for going from world->camera, etc.
    # We want "world->camera" transform, so if 'world_mat_inv' goes from world->camera,
    # we can just use that directly for points in world space.
    world_to_cam = cameras[f'world_mat_inv_{view_idx}']

    # ----------------------- Load pointcloud ---------------------
    pc_np = np.load(pointcloud_npz_path)
    points_local = torch.tensor(pc_np['points'], dtype=torch.float32, device=device)  # (N, 3)
    loc = torch.tensor(pc_np['loc'], dtype=torch.float32, device=device)              # (3,)
    scale = torch.tensor(pc_np['scale'], dtype=torch.float32, device=device)          # scalar or (3,)

    # Convert local -> world. (Double-check your dataset specifics.)
    # e.g.: X_world = X_local * scale + loc
    # If scale is just a float, broadcast; if it’s shape (3,), do elementwise multiply.
    points_world = points_local * scale + loc  

    # Now transform world -> camera coordinates
    # We'll define a small function for homogeneous transform:
    def transform_points(pts, mat4x4):
        N = pts.shape[0]
        pts_hom = torch.cat([pts, torch.ones(N, 1, device=device)], dim=-1)  # (N,4)
        pts_cam = pts_hom @ mat4x4.transpose(0,1)                            # (N,4)
        return pts_cam[:, :3]   # drop homogeneous component

    points_cam = transform_points(points_world, world_to_cam)  # (N, 3)

    # If camera_mat_inv_{view_idx} is also relevant, combine it if necessary:
    # e.g. if your pipeline expects: camera_coords = (camera_mat_inv_{view} * X_cam)
    # but often 'world_mat_inv_{view}' already includes camera_mat_inv.  
    # (Verify in your code which combination is correct.)

    # ----------------- Project to normalized image plane -----------
    # For perspective cameras typically: x_ndc = X_cam / Z_cam, etc.
    # But it depends on your intrinsics and your definitions.
    # We'll do a naive version if camera_mat_inv is a simple f perspective:
    # *Be sure to replicate exactly how you formed [-1,1]^2 in your ray code*
    
    # Example: If X_cam is (x, y, z), then pixel coords in NDC might be (x/z, y/z, 1)
    # if your code used "cur_rays = [x, y, 1]" in camera space. 
    # Here’s a hypothetical:
    valid = points_cam[:, 2] > 0      # in front of camera
    points_cam_valid = points_cam[valid]
    
    x_ndc = points_cam_valid[:,0] / points_cam_valid[:,2]
    y_ndc = points_cam_valid[:,1] / points_cam_valid[:,2]

    # We now have approx (x_ndc, y_ndc) in [-1,1], ignoring any focal scaling.
    # If you actually used focal lengths, you need to incorporate them. 
    # This is just an example.

    # ----------------- Compare to your ray directions ---------------
    # For each valid point, we can find which pixel in your HxW is "closest" to (x_ndc, y_ndc).
    # Then check if the direction stored in ray_grid[view_idx, ..., 3:6] 
    # is roughly pointing from the camera origin to that point.

    # Convert ndc -> pixel indices in [0..W-1, 0..H-1]
    # e.g. x_pixel = ( (x_ndc + 1) / 2 ) * (W-1)
    #      y_pixel = ( (y_ndc + 1) / 2 ) * (H-1)
    x_pix = ( (x_ndc + 1) / 2 ) * (W - 1)
    y_pix = ( (y_ndc + 1) / 2 ) * (H - 1)

    # Round to nearest integer pixel
    x_pix_int = x_pix.round().long()
    y_pix_int = y_pix.round().long()

    # Mask out those that fall outside [0..W-1], [0..H-1]
    inside = (x_pix_int >= 0) & (x_pix_int < W) & (y_pix_int >= 0) & (y_pix_int < H)
    x_pix_int = x_pix_int[inside]
    y_pix_int = y_pix_int[inside]
    pts_3d_valid = points_cam_valid[inside]

    # Ray data from your grid
    #   ray_grid has shape (24, H, W, 6) -> [o_x, o_y, o_z, d_x, d_y, d_z]
    #   for a single view
    origins = ray_grid[view_idx, ..., 0:3]  # (H, W, 3)
    dirs    = ray_grid[view_idx, ..., 3:6]  # (H, W, 3)

    # We'll pick each pixel, get o, d
    # Then compute the 3D point = o + t*d that should match the point cloud in camera coords.
    # Because we have pts_3d_valid in camera coords too.
    # In camera coords, origin is typically (0, 0, 0) if world_mat_inv includes all transforms.
    # But let's not assume that. Let's do the real intersection:

    # For each pixel that has an associated point:
    check_errors = []
    for i in range(x_pix_int.shape[0]):
        px = x_pix_int[i]
        py = y_pix_int[i]
        pt_cam = pts_3d_valid[i]   # (3,)

        # The stored origin, direction:
        o = origins[py, px]   # (3,)
        d = dirs[py, px]      # (3,)

        # If in the same camera coordinate system, we want a scalar t s.t. o + t*d = pt_cam
        # Solve for t: t = (pt_cam - o) dot d / (d dot d), if directions are unit, d dot d ~ 1
        # This is just 1D. We'll do a quick check:
        numerator = (pt_cam - o).dot(d)
        denom     = d.dot(d)
        t         = numerator / denom

        # Then compare o + t*d with pt_cam
        pt_recon = o + t*d
        error = torch.norm(pt_recon - pt_cam)  # Euclidean distance
        print(error)
        check_errors.append(error.item())

    if len(check_errors) == 0:
        print("No valid points could be mapped to the image plane. Possibly an alignment issue.")
        return

    mean_error = np.mean(check_errors)
    std_error  = np.std(check_errors)
    print(f"[View {view_idx}] Average 3D reprojection error: {mean_error:.4f} +/- {std_error:.4f}")     
          
          
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