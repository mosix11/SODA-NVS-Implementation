import torch
import numpy as np

# Code based on https://github.com/stelzner/srt/blob/main/srt/data/nmr.py and generated by ChatGPT
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

