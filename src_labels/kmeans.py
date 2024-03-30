import torch 
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from torch.distributions.categorical import Categorical

from accelerate import Accelerator


class KMeans:

    def __init__(self,
                 centroids: Optional[Tensor] = None,
                 accelerator: Optional[Accelerator] = None) -> None:
                
        """
            Args:
                centroids (Tensor, Optional): the initial centroids.
                acceleator (Accelerator, Optional): the accelerator of the process.
        """

        self.accelerator = accelerator

        # default values to be used if accelerator is None
        self.device, self.num_processes, self.process_index = "cpu", 1, 0

        # default print function if accelerator is None
        self.print = print

        if self.accelerator is not None:
            self.device = accelerator.device 
            self.num_processes = accelerator.num_processes 
            self.process_index = accelerator.process_index
            self.print = accelerator.print

        self.centroids  = None
        self.n_clusters = None
        self.init(centroids)


    def init(self, centroids: Optional[Tensor] = None):
        """
            Initialize the clusters.
            Args:
                centroids (Tensor, Optional): the initial centroids.
        """
        self.centroids = centroids
        self.n_clusters = None if centroids is None else centroids.shape[0]


    @staticmethod
    def _closest_euclidean_dist_sq(X: Tensor, centroids: Tensor):
        """
            Returns the euclidean distance squares from any point to the closest centroid.
            Args:
                X (Tensor): tensor of shape [n_points, dimensions]
                centroids (Tensor): tensor of shape [n_centroids, dimensions]
            Returns:
                a tensor of shape [n_points] with the distances squared to the closest
                centroid.
        """

        distances = torch.cdist(X, centroids)
        distances, _ = torch.min(distances, dim=1)
        distances = distances ** 2
        return distances

    
    def predict(self, 
                X: Tensor, 
                cosine: Optional[bool] = True):

        """
            Predict the class for each vector in input matrix.
            Args:
                X (torch.Tensor): tensor of shape [n_points, n_dimensions]
               
            Returns:
                The predicted class for each input point. Additionaly the distances
                if return_distances == True.
        """

        # centroids on local rank
        centroids = self.centroids.clone().to(self.device)

        # chunk of points on local rank
        chunks = X.chunk(self.num_processes, dim=0)
        sizes = [len(c) for c in chunks]
        cumul_sizes = [sum(sizes[0:i]) for i in range(len(sizes))]
        current_rank_idx_from = cumul_sizes[self.process_index]

        indices  = torch.zeros(size=(len(X),), dtype=torch.int64).to(self.device)

        X = chunks[self.process_index].to(self.device)

        if cosine:
            X = F.normalize(X, dim=1)
            centroids = F.normalize(centroids, dim=1)

        distances = torch.cdist(X, centroids)
        distances, indices_cr = torch.min(distances, dim=1)

        indices[current_rank_idx_from:current_rank_idx_from+len(indices_cr)] = indices_cr

        if self.accelerator is not None and self.accelerator.num_processes > 1:
            indices = self.accelerator.reduce(indices)

        return indices
    

    def fit_predict(self,            
                    X: Tensor, 
                    stop_threshold: Optional[float] = 1e-8, 
                    max_iter: Optional[int] = 1000,
                    cosine: Optional[bool] = True,
                    verbose: Optional[bool] = True):
        """
            Fit the data. Initial centroids are changed with kmenas iterations.
            Finally the predicted cluster indices (one for each input point) are returned.
            Args:
                X (Tensor): a tensor of shape [n_points, dim]
                stop_threshold (float, Optional): if the distance of centroids is lower
                than this threshold, than the algorithm stops.
                max_iters (int, Optional): maximum number of iterations.
                cosine (bool, Optional): true to use cosine similarity.
                verbose (bool, Optional): to print information.
                
            Returns:
                the cluster assignments.
        """
        self.fit(X, stop_threshold, max_iter, cosine, verbose)
        return self.predict(X)


    @staticmethod
    def _bincount(indices, n_clusters):
        """
            Deterministic implementation of bincount. It counts the occurences of input.
        """
        result = torch.zeros(n_clusters).to(indices.device)

        for i in range(len(result)):
            result[i] = (indices == i).sum(dim=0)

        return result


    def fit(self, 
            X: Tensor, 
            stop_threshold: Optional[float] = 1e-8, 
            max_iter: Optional[int] = 1000,
            cosine: Optional[bool] = True,
            verbose: Optional[bool] = True):

        """
            Fit the data. Initial centroids are changed with kmeans iterations.
            Args:
                X (Tensor): a tensor of shape [n_points, dim]
                stop_threshold (float, Optional): if the distance of centroids is lower
                than this threshold, than the algorithm stops.
                max_iters (int, Optional): maximum number of iterations.
                cosine (bool, Optional): true to use cosine similarity.
                verbose (bool, Optional): to print information.
            Returns:
                the new centroids.
        """

        # centroids on local rank
        old_centroids = self.centroids.to(self.device)

        X = torch.tensor_split(X, self.num_processes)[self.process_index].to(self.device)

        if cosine:
            X = F.normalize(X, dim=1)
            old_centroids = F.normalize(old_centroids, dim=1)

        for i in range(max_iter):
            
            new_centroids = torch.zeros_like(old_centroids)
            distances = torch.cdist(X, old_centroids)

            # assign vectors to closest centroids
            distances, indices = torch.min(distances, dim=1)

            loss = distances.mean().item()

            samples_per_cluster = self._bincount(indices, self.n_clusters)
            new_centroids = new_centroids.index_add(0, indices, X)

            # reduce across processes
            if self.accelerator is not None and self.accelerator.num_processes > 1:
                new_centroids = self.accelerator.reduce(new_centroids)
                samples_per_cluster = self.accelerator.reduce(samples_per_cluster)
            
            #print(samples_per_cluster)

            for j in range(len(samples_per_cluster)):
                if samples_per_cluster[j] == 0:
                    self.print(f"Warning! Cluster {j} has zero elements!")
                    new_centroids[j] = old_centroids[j]
                    samples_per_cluster[j] = 1

            new_centroids = (new_centroids.T/samples_per_cluster).T
            
            if cosine:
                new_centroids = F.normalize(new_centroids, dim=1)
                update_distance = (1 - torch.sum(old_centroids * new_centroids, dim=1))/2
            else:
                update_distance = torch.sum((old_centroids - new_centroids)**2, dim=1)**0.5

            update_distance = update_distance.mean().item()

            old_centroids = new_centroids

            if verbose:
                self.print(f"KMEANS {i+1}/{max_iter}: update {update_distance}"+ 
                           f"(threshold = {stop_threshold})")

            if update_distance < stop_threshold:
                if verbose: self.print("Threshold reached: STOPPING!")
                break 

        self.centroids = new_centroids

        return self.centroids, loss
    



    @staticmethod
    def centroids_from_labels(X, Y) -> Tensor:
        """
            Computes the centroids of the clusters given vectors and assignments.
            
            Args:
                X (torch.Tensor): tensor of shape [n_points, dim], each row is a vector.
                Y (torch.Tensor): tensor of shape [n_points] of integers. Element at 
                index i is the label of row i of X (cluster *hard* assignment.)
            Returns:
                A matrix containing, in each row, a centroid. 
        """
        Y = nn.functional.one_hot(Y).float()
        return ((Y.T @ X).T/(torch.sum(Y, dim=0))).T  



    @staticmethod
    def centroids_from_logits(X, L, soft=False) -> Tensor:
        """
            Computes the centroids of the clusters given the logits of a predictor.
            
            Args:
                X (torch.Tensot): tensor of shape [n_points, dim], each row is a vector.
                L (torch.Tensor): tensor of shape [n_points, n_classes] of floats. Each 
                row presents the logits of the given sample.
                soft (bool): True to use soft average (using softmax), false to use hard
                assignment (argmax).
            Returns:
                A matrix containing, in each row, a centroid. 
        """

        if soft:
            L = nn.Softmax(dim=1)(L)
            return ((L.T @ X).T/(torch.sum(L, dim=0))).T
        else:
            _, Y = torch.max(L, dim=1)
            return KMeans.centroids_from_labels(X, Y)



    @staticmethod
    def centroids_from_probabilities(X, P) -> Tensor:
        """
            Computes the centroids of the clusters given the probabilities of a predictor.
            
            Args:
                X (torch.Tensor): tensor of shape [n_points, dim], each row is a vector.
                L (torch.Tensor): tensor of shape [n_points, n_classes] of floats. Each 
                row presents the logits of the given sample.
                soft (bool): True to use soft average (using softmax), false to use hard
                assignment (argmax).
            Returns:
                A matrix containing, in each row, a centroid. 
        """

        return ((P.T @ X).T/(torch.sum(P, dim=0))).T
    

    @staticmethod
    def centroids_kmeans_plusplus(X: torch.Tensor, 
                                  n_clusters: int, 
                                  accelerator: Accelerator = None,
                                  return_indices: bool = False):
        """
            Returns the initialization centroids based on kmeans++. 
            The first centroid is selected randomly among the points. 
            The second centroid is sampled from the points with probability 
            proportional to the squared distance to the closest centroid.
            The other centroids are sampled in the same way.
        """

        device = "cpu" if accelerator is None else accelerator.device
        n_samples, n_features = X.shape

        if accelerator is None or accelerator.is_main_process:
            X = X.to(device)

            # initialize centers and indices tensors
            centers = torch.empty((n_clusters, n_features), dtype=X.dtype, device=device)
            indices = -torch.ones(n_clusters, dtype=int, device=device)

            # sample the first centroid and save it
            first_centroid_idx = torch.randint(0, n_samples, (1,)).item()
            centers[0] = X[first_centroid_idx]
            indices[0] = first_centroid_idx

            # compute the distances between al the points and the first centroid.
            closest_distances = KMeans._closest_euclidean_dist_sq(X, centers[0].unsqueeze(0))

            # sample the other centroids
            for c in range(1, n_clusters):
                categorical = Categorical(probs=closest_distances)
                index = categorical.sample().item()
                centers[c] = X[index]
                indices[c] = index

                # update the distances
                closest_distances = KMeans._closest_euclidean_dist_sq(X, centers[0:c+1])
        else:
            
            centers = torch.zeros((n_clusters, n_features), dtype=X.dtype, device=device)
            indices = torch.zeros(n_clusters, dtype=int, device=device)

        if accelerator is not None and accelerator.num_processes > 1:
            centers = accelerator.reduce(centers)
            indices = accelerator.reduce(indices)

        if return_indices:
            return centers, indices
        
        return centers