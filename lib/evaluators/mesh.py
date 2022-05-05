import numpy as np
import open3d as o3d
from sklearn.neighbors import KDTree


def nn_correspondance(verts1, verts2):
    indices = []
    distances = []
    if len(verts1) == 0 or len(verts2) == 0:
        return indices, distances

    kdtree = KDTree(verts1)
    distances, indices = kdtree.query(verts2)
    distances = distances.reshape(-1)

    return distances


class Evaluator:
    def evaluate(self, mesh_pred, mesh_trgt, threshold=.05, down_sample=.02):
        pcd_pred = o3d.geometry.PointCloud(mesh_pred.vertices)
        pcd_trgt = o3d.geometry.PointCloud(mesh_trgt.vertices)

        if down_sample:
            pcd_pred = pcd_pred.voxel_down_sample(down_sample)
            pcd_trgt = pcd_trgt.voxel_down_sample(down_sample)

        verts_pred = np.asarray(pcd_pred.points)
        verts_trgt = np.asarray(pcd_trgt.points)

        dist1 = nn_correspondance(verts_pred, verts_trgt)
        dist2 = nn_correspondance(verts_trgt, verts_pred)

        precision = np.mean((dist2 < threshold).astype('float'))
        recal = np.mean((dist1 < threshold).astype('float'))
        fscore = 2 * precision * recal / (precision + recal)
        metrics = {
            'Acc': np.mean(dist2),
            'Comp': np.mean(dist1),
            'Prec': precision,
            'Recal': recal,
            'F-score': fscore,
        }
        return metrics
