import numpy as np

# internal imports
from src import config


def get_agent_track(scene, mode="train"):
    """Returns target agent trajectory"""
    agent_idx = np.where(scene["agent_id"] == np.unique(scene["track_id"].flatten()))[0][0]
    agent_traj = scene['p_in'][agent_idx]
    if mode == "test":
        return agent_traj
    else:
        agent_traj = np.concatenate([agent_traj, scene['p_out'][agent_idx]])
    return agent_traj


def get_social_tracks(scene, mode="train"):
    """Returns other agent trajectory"""
    agent_idx = np.where(scene["agent_id"] == np.unique(scene["track_id"].flatten()))[0][0]
    social_masks = scene["car_mask"].flatten()
    social_masks[agent_idx] = 0
    social_trajs = scene['p_in'][social_masks.astype(bool)]
    
    if mode == "test":
        return social_trajs
    else:
        return np.concatenate([social_trajs, scene['p_out'][social_masks.astype(bool)]], axis=1)

def count_num_neighbors(agent_traj: np.ndarray, social_trajs: np.ndarray):
    """
    Calculate euclidean distance between agent_traj and social_trajs
    if distance is less than NEARBY_DISTANCE_THRESHOLD, then num_neighbors++
    
    Args:
        agent_traj (np.ndarray): data for agent trajectory
        social_trajs (np.ndarray): array of other agents' trajectories
    Returns:
        (np.array): 
    """
    num_neighbors = []
    dist = np.sqrt(
        (social_trajs[:, :, 0] - agent_traj[:, 0])**2 
        + (social_trajs[:, :, 1] - agent_traj[:, 1])**2
    ).T
    num_neighbors = np.sum(dist < config.NEARBY_DISTANCE_THRESHOLD, axis=1)
    return num_neighbors.reshape(-1, 1)
    
    
def get_social_features(scene: dict, mode: str="train"):
    """
    Compute social features:
        - Number of neighbors
    
    Args:
        scene (dict): raw scene data
        mode (str): train/test 
    Returns:
        social_features (np.ndarray): (seq_len x num_features) social features for the target agent
    """
    agent_track = get_agent_track(scene, mode)
    social_tracks = get_social_tracks(scene, mode)
    # compute social features
    num_neighbors = count_num_neighbors(agent_track[:19], social_tracks[:, :19])
    return num_neighbors
    