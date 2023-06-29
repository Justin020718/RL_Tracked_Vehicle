import numpy as np
from sklearn.neighbors import NearestNeighbors


def best_fit_transform(A, B):
    '''
    Calculates the least-squares best-fit transform that maps corresponding points A to B in m spatial dimensions
    Input:
      A: Nxm numpy array of corresponding points
      B: Nxm numpy array of corresponding points
    Returns:
      T: (m+1)x(m+1) homogeneous transformation matrix that maps A on to B
      R: mxm rotation matrix
      t: mx1 translation vector
    '''

    assert A.shape == B.shape #判断A和B的形状

    # get number of dimensions
    m = A.shape[1]#取A的列,3

    # translate points to their centroids
    centroid_A = np.mean(A, axis=0) #计算质心
    centroid_B = np.mean(B, axis=0) #计算质心
    AA = A - centroid_A #转换到质心
    BB = B - centroid_B #转换到质心

    # rotation matrix
    H = np.dot(AA.T, BB) #构造H
    U, S, Vt = np.linalg.svd(H) #SVD分解
    R = np.dot(Vt.T, U.T) #得到旋转矩阵

    # special reflection case
    if np.linalg.det(R) < 0:
       Vt[m-1,:] *= -1
       R = np.dot(Vt.T, U.T)

    # translation
    t = centroid_B.T - np.dot(R,centroid_A.T)#平移矩阵

    # homogeneous transformation
    T = np.identity(m+1) #构造变换矩阵4*4
    T[:m, :m] = R #旋转矩阵放于左上角3*3
    T[:m, m] = t #平移矩阵放置右上角3*1

    return T, R, t


def nearest_neighbor(src, dst):#src源、dst目标，目的采用K近邻方法寻找对应点
    '''
    Find the nearest (Euclidean) neighbor in dst for each point in src
    Input:
        src: Nxm array of points
        dst: Nxm array of points
    Output:
        distances: Euclidean distances of the nearest neighbor
        indices: dst indices of the nearest neighbor
    '''

    assert src.shape == dst.shape

    neigh = NearestNeighbors(n_neighbors=1) #非监督的最近邻学习，近邻点设为1
    neigh.fit(dst) #适配到目标
    distances, indices = neigh.kneighbors(src, return_distance=True) #找到一个点的一个邻居，返回每个点的邻居的索引和距离。
    return distances.ravel(), indices.ravel()


def icp(A, B, init_pose=None, max_iterations=20, tolerance=0.000001):
    '''
    The Iterative Closest Point method: finds best-fit transform that maps points A on to points B
    Input:
        A: Nxm numpy array of source mD points
        B: Nxm numpy array of destination mD point
        init_pose: (m+1)x(m+1) homogeneous transformation
        max_iterations: exit algorithm after max_iterations
        tolerance: convergence criteria
    Output:
        T: final homogeneous transformation that maps A on to B
        distances: Euclidean distances (errors) of the nearest neighbor
        i: number of iterations to converge
    '''

    assert A.shape == B.shape #同上

    # get number of dimensions
    m = A.shape[1] #同上

    # make points homogeneous, copy them to maintain the originals
    src = np.ones((m+1,A.shape[0])) #构造新的数组
    dst = np.ones((m+1,B.shape[0]))
    src[:m,:] = np.copy(A.T) #转置下面多了一行1，列数和原矩阵行数一样
    dst[:m,:] = np.copy(B.T)

    # apply the initial pose estimation
    if init_pose is not None: #判断是否有原有转换
        src = np.dot(init_pose, src) #按原有变换进行变换

    prev_error = 0

    for i in range(max_iterations): #设定最大迭代次数（后面也设定阈值作为迭代终点）
        # find the nearest neighbors between the current source and destination points
        distances, indices = nearest_neighbor(src[:m,:].T, dst[:m,:].T) #同英语，即寻找对应点

        # compute the transformation between the current source and nearest destination points
        T,_,_ = best_fit_transform(src[:m,:].T, dst[:m,indices].T) #计算变换矩阵，包括旋转及平移矩阵

        # update the current source
        src = np.dot(T, src) #按变换矩阵更新source

        # check error
        mean_error = np.mean(distances) #检查对应点偏差是否比设定偏差要小，小的话就结束了
        if np.abs(prev_error - mean_error) < tolerance:
            break
        prev_error = mean_error

    # calculate final transformation
    T,_,_ = best_fit_transform(A, src[:m,:].T) #完成迭代，得到变换矩阵

    return T, distances, i #返回变换矩阵、最小距离及迭代次数