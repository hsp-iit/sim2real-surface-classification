from fire import Fire
from pc import PointCloud
from os import listdir
from os.path import join, isdir
import numpy as np



####################### UTIL FUNCTIONS #################################################

def find_class(curvature: float, thresholds: list):
    """ 
        Given the curvature curvature and a list of thresholds it outputs the class of 
        the point
    """

    found_class = 0     
    for t in sorted(thresholds):
        if curvature > t:
            found_class += 1
        else:
            return found_class
    return found_class

def list2csv(values: list):
    """
        Converts a list of values to a comma separated string. 
    """
    values = list(map(lambda x: str(x), values))
    return ",".join(values)

def read_csv(path, skiplines=1):
    with open(path, "r") as f:
        lines = [l.rstrip() for l in f.readlines()]

    lines = lines[skiplines:]
    lines = [list(map(lambda x: float(x), l.split(","))) for l in lines]
    return lines
########################################################################################


def compute_curvatures(root: str,
                       object_filename: str = "poisson_disk_01.ply",
                       curvature_file_out: str = "CURVATURES.csv",
                       radius: float = 0.006,
                       max_angle_delta: float = 150.,
                       verbose: bool = True):
    """
        Compute the curvatures of all objects in root folder and save them in a csv file.

        Args:
            root (str): the folder containing the objects (each object should be in a 
            sub-folder).
            object_filename (str): name of the ply file of each object.
            curvature_file_out (str): output file with curvatures.
            radius (float): radius to compute neighborhood of points.
            max_angle_delta (float): maximum angle between normals of points in the
            neighborhood.
            verbose (bool): True to print info.
    """

    # all subdirs of root dir
    dirs = [r for r in listdir(root) if isdir(join(root, r))]

    for i, dir in enumerate(dirs):

        if verbose: print(f"COMPUTING CURVATURE {i+1}/{len(dirs)} ({dir})")

        # load the pointcloud
        pc = PointCloud(path=join(root, dir, object_filename))

        # get curvatures
        curvatures, points, normals = pc.compute_curvature(radius=radius, 
                                                           max_angle_delta=max_angle_delta)
        out_path = join(root, dir, curvature_file_out)

        with open(out_path, "w+") as f:

            print("point_x,point_y,point_z,normal_x,normal_y,normal_z,curvature", file=f)
            for c, p, n in zip(curvatures, points, normals):
                values = list(p) + list(n) + [c] 
                print(list2csv(values), file=f)

def compute_labels(root: str,
                   curvature_file: str = "CURVATURES.csv",
                   labels_out_file: str = "LABELS.csv",
                   thresholds: list = [0.28, 0.48],
                   exp: float = 0.2,
                   verbose: bool = True):

    """
        Compute 

        Args:
            root (str): the folder containing the objects (each object should be in a 
            sub-folder).
            curvature_file (str): input file with curvatures.
            labels_out_file (str): output files with labels.
            thresholds (list): list of floats to convert curvatures into labels.
            exp (float): exponent to apply to curvature before the use of thresholds.
            verbose (bool): True to print info.

    """
        
    # all subdirs of root dir.
    dirs = [r for r in listdir(root) if isdir(join(root, r))]

    for i, dir in enumerate(dirs):

        if verbose: print(f"COMPUTING LABELS {i+1}/{len(dirs)} ({dir})")

        curvatures_file = join(root, dir, curvature_file)
        values = np.array(read_csv(curvatures_file, skiplines=1))

        points     = values[:, 0:3]
        normals    = values[:, 3:6]
        curvatures = values[:, 6] ** exp
        labels     = [find_class(c, thresholds) for c in curvatures]

        labels_out_path = join(root, dir, labels_out_file)

        with open(labels_out_path, "w+") as f:

            print("point_x,point_y,point_z,normal_x,normal_y,normal_z,curvature_exp,label", file=f)
            for p,n,c,l in zip(points, normals,curvatures, labels):
                values = list(p) + list(n) + [c,l] 
                print(list2csv(values), file=f)


def filter_curve_points(root: str,
                        labels_file: str = "LABELS.csv",
                        labels_filtered_file_out: str = "LABELS_FILTERED.csv",
                        radius: float = 0.006,
                        verbose: bool = True):
    """
        Filter curve point between flat and hard-curve points.  

        Args:
            root (str): the folder containing the objects (each object should be in a 
            sub-folder).
            labels_file (str): the input label files (output of compute_labels function).
            labels_filtered_file_out (str): the output file.
            radius (str): radius to compute the neighborhood of points.
            verbose (bool): True to print info.
    """

    # all subdirs of root dir.
    dirs = [r for r in listdir(root) if isdir(join(root, r))]

    ## FILTER CURVE POINTS
    for i, dir in enumerate(dirs):

        if verbose: print(f"Filtering {i+1}/{len(dirs)} ({dir})")

        current_labels_file = join(root, dir, labels_file)
        pc = PointCloud(current_labels_file)
        points  = pc.points
        normals = pc.normals
        CURVE_CLASS = 1

        for _ in range(2):
            labels  = pc.labels
            curve_idx = (labels==CURVE_CLASS).nonzero()[0]
        
            for point_idx in curve_idx:
                output_label = pc.filter_curve(radius=radius, idx=point_idx)
                pc.labels[point_idx] = output_label
        

        labels_filtered_file = join(root, dir, labels_filtered_file_out)

        labels = pc.labels

        with open(labels_filtered_file, "w+") as f:
            print("point_x,point_y,point_z,normal_x,normal_y,normal_z,label", file=f)
            for p,n,l in zip(points, normals, labels):
                values = list(p) + list(n) + [l] 
                print(list2csv(values), file=f)




## k23-loss and normal angles
def compute_k23_loss(root: str,
                     labels_filtered_file: str = "LABELS_FILTERED.csv",
                     k23_file_out: str = "K23.csv",
                     thresholds: list = [0.28, 0.48],
                     radius: float = 0.008,
                     
                     verbose: bool = True):
    """
        Compute the k23 loss: K-Means is applied on the normals of neighbour points with
        K=2 and K=3 and the difference of the two losses is saved. Additionally the 
        min and max angles between the normals of K=3 centroids are saved to enable a 
        the selection of corner points.

        Args:
            root (str): the folder containing the objects (each object should be in a 
            sub-folder).
            labels_filtered_file (str): file with the labels filtered
              (out of filter_curve_points function).
            k23_file_out (str): the output file.
            thresholds (list): the list of thresholds used to split curves, used to infer
            the number of classes.
            radius (float): radius to compute the neighborhood of points.
            verbose (bool): True to print info.
    """
        
    # all subdirs of root dir.
    dirs = [r for r in listdir(root) if isdir(join(root, r))]

    for i, dir in enumerate(dirs):

        if verbose: print(f"COMPUTING k23 loss and normal angles {i+1}/{len(dirs)} ({dir})")

        labels_file = join(root, dir, labels_filtered_file)
        pc = PointCloud(labels_file)
        points  = pc.points
        normals = pc.normals
        labels  = pc.labels

        n_points = len(points)

        # number of classes = # threshols + 1 (from 0 to len(thressholds). Edge class is the
        # last one i.e. len(thresholds) 
        edge_class = len(thresholds) 

        # get just points of the edge class
        edges_idx = (labels==edge_class).nonzero()[0]
        
        # initialize 
        k23_losses_array = -np.ones(n_points)
        angles_array     = np.zeros([n_points, 3])

        for point_idx in edges_idx:
            
            k23_loss, angles, convex = pc.compute_23_loss_angles(radius=radius, idx=point_idx)
            k23_losses_array[point_idx] = k23_loss
            angles_array[point_idx] = angles

        
        k23_file = join(root, dir, k23_file_out)

        with open(k23_file, "w+") as f:

            print("point_x,point_y,point_z,normal_x,normal_y,normal_z, k23_loss, "
                +"theta0, theta1, theta2, label", file=f)
            for p,n,loss,angles,l in zip(points, normals,k23_losses_array, angles_array, labels):
                values = list(p) + list(n) + [loss] + list(angles) + [l]
                print(list2csv(values), file=f)


def get_corners(root: str, 
                k23_file: str = "K23.csv",
                corner_file_out: str ="LABELS_CORNERS.csv",
                thresholds: list = [0.28, 0.48],
                min_loss_delta: float = 0.0004,
                min_angle: float = 60,
                max_angle: float = 120,
                verbose: bool = True):

    """
        Compute the corners from K23 loss file. 

        Args:
            root (str): the folder containing the objects (each object should be in a 
            sub-folder).
            k23_file (str): the name of the k23_loss file.
            corner_file_out (str): output file with corners.
            thresholds (list): the threshold used to split curvatures (used to infer the
            number of classes).
            min_loss_delta (float): minumum delta of the k23 loss for corners.
            min_angle (float): minimin angle between the normals of centroids (found with 
            K-Means=3) to consider corners.
            max_angle (float): maximum angle between the normals of centroids (found with 
            K-Means=3) to consider corners.
            verbose (bool): to to print infos.
    """
        
    # all subdirs of root dir.
    dirs = [r for r in listdir(root) if isdir(join(root, r))]

    for i, dir in enumerate(dirs):

        if verbose: print(f"SAVING CORNER LABELS {i+1}/{len(dirs)} ({dir})")

        labels_file = join(root, dir, k23_file)

        with open(labels_file, "r") as f:

            lines = [l.rstrip().split(",") for l in f.readlines()][1:]
            points  = np.array([[float(l[0]), float(l[1]), float(l[2])] for l in lines])
            normals = np.array([[float(l[3]), float(l[4]), float(l[5])] for l in lines])
            k23_losses = np.array([float(l[6]) for l in lines])
            min_angles = np.array([float(l[7]) for l in lines])
            max_angles = np.array([float(l[9]) for l in lines])
            labels = np.array([int(l[10]) for l in lines])


        with open(join(root, dir, corner_file_out), "w+") as f:

            print("point_x,point_y,point_z,normal_x,normal_y,normal_z,label", file=f)
            
            for p,n,loss,mi,ma,l in zip(points, normals, k23_losses, min_angles, max_angles, labels):
                
                corner_label = len(thresholds) + 1 
                if loss > min_loss_delta and abs(mi) > min_angle and abs(ma) < max_angle: l = corner_label

                values = list(p) + list(n) + [l]
                print(list2csv(values), file=f)


if __name__ == "__main__":

    root = "input_objects"

    compute_curvatures(root)
    compute_labels(root)
    filter_curve_points(root)
    compute_k23_loss(root)
    get_corners(root)