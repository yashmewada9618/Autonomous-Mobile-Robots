import numpy as np
import time
import matplotlib.pyplot as plt

class IterativeClosedPoint:
    def __init__(self,X,Y):
        """
        Constructor Method
        Args:
            X: X point cloud. This is basically your moving point cloud
            Y: Y point cloud. This is basically your fixed point cloud
        """
        self.X = X
        self.Y = Y
        self.xlist = []
        self.ylist = []
        self.W = np.matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

    def EstimateCorrespondences(self,t,R,dmax):
        """
        Args:
            t: Optimal translation you found using SVD decomposition. For initial guess we consider it as 0.
            R: Optimal rotation you found using SVD decomposition. For initial guess we consider it as I3 i.e no roation is there between the points.
        Returns:
            c: A list of tuple with indices of the corresponded X and Y points from the point clouds.
        """
        c = []
        x = np.transpose(self.X) # Transpose the point clouds for easy matrix calculations
        y = np.transpose(self.Y)
        for i in range(self.X.shape[0]):
            xp = x[:,i]
            yj = np.linalg.norm(y - (np.dot(R,xp) + t),axis=0) #calculate the norm/ euclidean distance between y and R.x+t
            yind = np.argmin(yj) # get the index of the minimum norm from those calculated
            if (yj[yind] < dmax):
                c.append((i,yind)) # append those indices to the corressponded list.
                self.xlist.append(self.X[i])
                self.ylist.append(self.Y[yind])
        return c

    def ComputeOptimalRigidRegistration(self,corresponds):
        """
        Args:
            corresponds : The list of corresponded points you found from EstimateCorrespondences function
        Returns:
            trans: Optimal translation from the corresponded points
            rot: Optimal rotation from the corresponded points
        """
        Xcentroid = np.matrix([0.,0.,0.]) #declare empty float matrices for centroids
        Ycentroid = np.matrix([0.,0.,0.])

        Xcentroid = np.mean(self.xlist,axis=0) #calculate mean of teh found corresponded points from the cloud
        Ycentroid = np.mean(self.ylist,axis=0)

        sdx = np.subtract(self.xlist,Xcentroid) # caluculate their deviation from centroid
        sdy = np.subtract(self.ylist,Ycentroid)

        for i in range(len(corresponds)):
            self.W += np.dot(sdy[i,:].T,sdx[i,:]) # calculate the cross covariance matrix
        self.W = self.W/len(corresponds)

        U,_,vt = np.linalg.svd(self.W) #decompse the found cross covariance matrix
        rot = np.dot(U,vt) #calculate rotation and optimal translation from SVD
        trans = Ycentroid.T - (np.dot(rot,Xcentroid.T))

        return (trans,rot) #return those translations and rotational matrices

    def compute(self,t,R,dmax,num_ICP_iters):
        """
        Args:
            t: Optimal translation you found using SVD decomposition. For initial guess we consider it as 0.
            R: Optimal rotation you found using SVD decomposition. For initial guess we consider it as I3 i.e no roation is there between the points.
            dmax: The maximum overlapping distance.
            num_ICP_iters: Number of iterations the ICP should run.
        Returns:
            tr: Optimal translation from the corresponded points
            ro: Optimal rotation from the corresponded points
            cor: A list of tuples of indices of final corresponded points.
        """
        tr = t
        ro = R
        for i in range(num_ICP_iters):
            cor = self.EstimateCorrespondences(tr,ro,dmax)
            tr,ro = self.ComputeOptimalRigidRegistration(cor)
            self.xlist = []
            self.ylist = []
            self.W = np.matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

            # USED FOR GENERATING GIF FROM EACH TRANSFORMATIONS BETWEEN THE CLOUDS
            # icx = np.dot(ro,self.X.T) + tr
            # icx = icx.T
            # fig = plt.figure()
            # ax = fig.add_subplot(111, projection='3d')
            # ax.scatter(self.Y[0:,0], self.Y[0:,1], self.Y[0:,2], c='r', marker='o',s=0.5)
            # ax.scatter(icx[0:,0], icx[0:,1], icx[0:,2], c='b', marker='^',s=0.5)
            # ax.set_xlabel('X Label')
            # ax.set_ylabel('Y Label')
            # ax.set_zlabel('Z Label')
            # filename = 'plot-' + str(i) +'.png'
            # fig.savefig(filename)
            # plt.close(fig)

        return tr,ro,cor
        
if __name__ == "__main__":
    x = np.loadtxt("C:/Users/yashm/Downloads/pclX.txt", dtype=float)
    xpcd = np.matrix(x)
    
    y = np.loadtxt("C:/Users/yashm/Downloads/pclY.txt", dtype=float)
    ypcd = np.matrix(y) 
    
    icp = IterativeClosedPoint(xpcd,ypcd)
    
    R = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    t = np.matrix([[0],[0],[0]])
    
    start = time.time()
    trans,rot,cor = icp.compute(t,R,0.25,30)
    print("The ICP took %s seconds to compute" % (time.time()-start))
    icx = np.dot(rot,xpcd.T) + trans
    
    RMSE = 0
    for i,j in cor:
        x = xpcd[i].T
        y = ypcd[j].T
        y_norm = np.linalg.norm(y - (np.dot(rot,x) + trans))
        RMSE+=np.power(y_norm,2)
    RMSE= np.sqrt(RMSE/len(cor))

    print("RMSE: ",RMSE)
    print("Translation Matrix: ",trans.T)
    print("Rotation Matrix: ",rot)
    print("Lenght of Correspondences: ",len(cor))

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(icx[0], icx[1], icx[2], c='r', marker='^',label = 'Moving',s = 0.5)
    ax.scatter(ypcd[:,0], ypcd[:,1], ypcd[:,2], c='b',marker='^',label = 'Fixed',s = 0.5)
    ax.legend()
    # ax.scatter(xpcd[:,0], xpcd[:,1], xpcd[:,2], c='g',s=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    plt.show()
