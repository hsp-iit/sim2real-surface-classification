import open3d as o3d
from open3d.utility import Vector3dVector
import numpy as np
import torch
from kmeans import KMeans
import torch.nn.functional as F
from sklearn.metrics.pairwise import cosine_similarity
from einops import repeat

from typing import Tuple


class PointCloud:

    def __init__(self, path: str):
        """ 
            Loads the point clound (and normals) from a file. Currently the class 
            supports ply for unlabeled point clouds and csv for labeled point clouds, 
            in particular:

            - ply: includes points coordinates and normals.
            - csv: comma separated values. Each row should have 7 values: 3 values for
                   xyz coordinates of the point, 3 values to specify the normal of the
                   point and 1 value to specify the label.

            Args:
                path (str): the path to the file to load.

            Raise:
                ValueError: if the file specified in not supported.

        """

        self.pcd     = None  
        self.points  = None
        self.normals = None
        self.labels  = None

        # load the ply point cloud
        if path[-4:] == ".ply":
            self.pcd     = o3d.io.read_point_cloud(path)
            self.points  = np.asarray(self.pcd.points)
            self.normals = np.asarray(self.pcd.normals)

            # normalize normals
            self.normals = self.normals / np.linalg.norm(self.normals)

        # load the csv labeled point cloud
        elif path[-4:] == ".csv":
            with open(path, "r") as f:
                lines = [l.rstrip().split(",") for l in f.readlines()][1:]

                # first 3 values are the xyz coordinates
                xyz = np.array([[float(l[0]), float(l[1]), float(l[2])] for l in lines])

                # following 3 values are the normals
                uvw = np.array([[float(l[3]), float(l[4]), float(l[5])] for l in lines])

                # 7th value is the label associated to each point
                self.labels = np.array([int(l[-1]) for l in lines])

                self.pcd = o3d.geometry.PointCloud()
                self.pcd.points = Vector3dVector(xyz)

                self.normals = uvw / np.linalg.norm(uvw) # normalize normals
                self.points  = xyz

        else:
            raise ValueError("File not supported.")

        self.tree = o3d.geometry.KDTreeFlann(self.pcd)

        

    def compute_curvature(self, 
                          radius: float,
                          max_angle_delta=150) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
            Args:
                radius (float): the radius to compute the curvature.
                max_angles_delta (float): maximum angle delta (in degrees) between the 
                normals of the points to be considered. If the angle of normals are 
                greater than this for some points, than such points will not be 
                considered in the computation of curvature. This is to prevent the use
                of points of the opposite surface when the obeject is thinner than the
                radius.

            Returns:
                a tuple with 3 np.array: curvatures, points and normals
        """

        points = np.asarray(self.pcd.points)
        kdtree = o3d.geometry.KDTreeFlann(self.pcd)

        curvatures  = []
        out_points  = []
        out_normals = []

        for i in range(len(points)):

            # get neighbours points (inside a sphere of given radius)
            _, idxs, _ = kdtree.search_radius_vector_3d(self.pcd.points[i], radius)

            idxs = np.array(idxs)

            current_normal     = self.normals[i, :][None, :]
            neighbours_normals = self.normals[idxs, :]
            
            similarities = cosine_similarity(neighbours_normals, current_normal)[:, 0]
            similarities[similarities >= 1.] = 1.

            # convert similarities to angles
            angles = np.arccos(similarities) * 180 / np.pi

            # filter points of opposite surface 
            # (i.e. with angles of normals >= max_angle_delta)

            idxs_ok = (angles < max_angle_delta).nonzero()[0]
            idxs = idxs[idxs_ok]

            neighbours_points = points[idxs, :]

            # center points
            neighbours_points -= np.mean(neighbours_points, axis=0) 

            # compute singular values
            cov_matrix = np.dot(neighbours_points.T, neighbours_points)
            _, sigmas, _ = np.linalg.svd(cov_matrix)

            # sort singular values (ascending)
            sigmas = np.sort(sigmas)

            # sum of singular values
            sum_sigma = np.sum(sigmas)
            smallest_sigma = sigmas[0]

            # curvature
            curvature = smallest_sigma/sum_sigma if sum_sigma != 0 else 0.

            curvatures.append(curvature)
            out_points.append(list(self.pcd.points[i]))
            out_normals.append(list(self.normals[i, :]))

        return np.array(curvatures), np.array(out_points), np.array(out_normals)
    

    def filter_curve(self, radius: float, idx: int):
        #kdtree = o3d.geometry.KDTreeFlann(self.pcd)
        
        _, neigh_idxs, _ = self.tree.search_radius_vector_3d(self.pcd.points[idx], radius)
        neight_labels = torch.tensor(self.labels[neigh_idxs])

        if torch.sum(neight_labels==0) > torch.sum(neight_labels==2) and torch.sum(neight_labels==2) > 5:
            return 0
        elif torch.sum(neight_labels==0) < torch.sum(neight_labels==2) and torch.sum(neight_labels==0) > 5:
            return 0
        else:
            return 1 
        

    def compute_23_loss_angles(self, 
                               radius: float, 
                               idx: int,
                               convex_th=82) -> Tuple:
        """ 
            This function is needed to discriminate between edges and corners:
            for a given point it extracts the neighbourhood, then it apply K-Means on the
            normals of these points with K=2 and K=3. The loss_k23 is the difference 
            between the KMeans losses with K=2 and K=3: a large value indicates a corner
            while a low value indicates an edge. Additionally the concave corners
            are filtered and they are assigned a 0 value of loss_k23. 

            Args:
                radius (float): radius of the neighboorhood to compute the k23 loss.
                idx (int): the idx of the considered point under examination.
                convex_th (float): threshold angle to remove concave angles and to keep 
                convex ones. 

            Returns:
                k23_loss (float), List of 3 angles between centroids (found with K=3) and
                boolean that is True for convex corners and false otherwise.

        """
        
        _, neigh_idxs, _ = self.tree.search_radius_vector_3d(self.pcd.points[idx], radius)
        neighbours_normals = torch.tensor(self.normals[neigh_idxs, :])
        neight_points      = torch.tensor(np.asarray(self.pcd.points)[neigh_idxs, :])
        current_point      = torch.tensor(self.pcd.points[idx])
        current_normal     = torch.tensor(self.normals[idx])

        diff_vectors =  neight_points - repeat(current_point, "c -> n c", n = len(neight_points))
        diff_vectors = F.normalize(diff_vectors, dim=1)
        current_normal = F.normalize(current_normal, dim=0)

        angles = current_normal @ diff_vectors.T
        angles = angles[1:]
        angles = torch.arccos(angles) * 180 / torch.pi

        # concave
        if torch.sum(angles < convex_th) > 0:
            return 0, [0,0,0], False
  
        # K = 2
        init_centroids = KMeans.centroids_kmeans_plusplus(X=neighbours_normals, 
                                                          n_clusters=2)
        
        cluster = KMeans(centroids=init_centroids)
        _, loss2 = cluster.fit(X=neighbours_normals, 
                               stop_threshold=1e-16, 
                               max_iter=50, 
                               cosine=False, 
                               verbose=False)
        
        # K = 3
        init_centroids = KMeans.centroids_kmeans_plusplus(X=neighbours_normals, 
                                                          n_clusters=3)
        
        cluster = KMeans(centroids=init_centroids)
        centroids3, loss3 = cluster.fit(X=neighbours_normals, 
                                        stop_threshold=1e-16, 
                                        max_iter=50, 
                                        cosine=False, 
                                        verbose=False)

        # ANGLES of K=3
        centroids3 = F.normalize(centroids3, dim=1)
        angles3 = centroids3 @ centroids3.T
        angles3 = [angles3[0,1], angles3[0, 2], angles3[1, 2]]
        angles3 = list(map(lambda x: torch.arccos(x).item() * 180 / torch.pi, angles3))

        return loss2-loss3, sorted(angles3), True
