import numpy as np
import time
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
class IterativeClosedPoint:
    def __init__(self,X,Y):
        self.X = X
        self.Y = Y
        self.C = []
        self.xlist = []
        self.ylist = []
        self.W = np.matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])

    def EstimateCorrespondences(self,t,R,dmax):
        c = []
        x = np.transpose(self.X) # X point cloud
        y = np.transpose(self.Y) # # X point cloud
        for i in range(self.X.shape[0]):
            xp = x[:,i]
            yj = np.linalg.norm(y - (np.dot(R,xp) + t),axis=0)
            yind = np.argmin(yj)
            if (yj[yind] < dmax):
                self.C.append((i,yind))
                # np.append()
                # self.xlist.append(self.X[i])
                # self.ylist.append(self.Y[yind])
        return self.C
    
    def plot2gif(X,Y,i):
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[0:,0], X[0:,1], X[0:,2], c='r', marker='o',s=0.5)
        ax.scatter(Y[0:,0], Y[0:,1], Y[0:,2], c='b', marker='^',s=0.5)
        ax.set_xlabel('X Label')
        ax.set_ylabel('Y Label')
        ax.set_zlabel('Z Label')
        filename = str(i)+'plot.png'
        fig.savefig(filename)
        plt.close(fig)
        return

    def ComputeOptimalRigidRegistration(self,corresponds):
        W = np.matrix([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
        # xl = [self.X[i] for i,_ in corresponds]
        # yl = [self.Y[j] for _,j in corresponds]
        # print(np.array(xl)[3299])
        # print(np.array(xl).shape)
        # self.xlist = [self.X[i] for i,_ in corresponds]
        # self.ylist = [self.Y[j] for _,j in corresponds]
        # newx = np.asarray(self.xlist).reshape(-1,3)
        # newy = np.asarray(self.ylist).reshape(-1,3)
        # print(newx.shape)
        Xcentroid = np.matrix([0.,0.,0.])
        Ycentroid = np.matrix([0.,0.,0.])
        for i,j in corresponds:
            Xcentroid += self.X[i]
            Ycentroid += self.Y[j]
        Xcentroid = Xcentroid/len(corresponds)
        Ycentroid = Ycentroid/len(corresponds)
        # Xcentroid = np.mean(self.xlist,axis=0)
        # Ycentroid = np.mean(self.ylist,axis=0)
        # print(Xcentroid)
        # print(Ycentroid)
        # Xcentroid = sum(self.xlist)/len(self.xlist)
        # Ycentroid = sum(self.ylist)/len(self.ylist)
        # sdx = np.subtract(self.xlist,Xcentroid)
        # sdy = np.subtract(self.ylist,Ycentroid)
        for i,j in corresponds:
            sdx = self.X[i] - Xcentroid
            sdy = self.Y[j] - Ycentroid
            # print(sdx.shape)
            W += np.dot(sdy.T,sdx)
        # print(np.array(self.xlist[0:]).shape)
        # for i in range(len(corresponds)):
        #     self.W += sdx[i,:].T@sdy[i,:]
            # self.W += np.outer(sdx[i,:],sdy[i,:].T)  
        W = W/len(corresponds)
        # print(self.W.shape)
        U,_,vt = np.linalg.svd(W)
        # s = np.identity(3)
        # s[2][2] = np.linalg.det(np.matmul(U,vt))
        # print(s)
        rot = np.dot(U,vt)
        trans = Ycentroid.T - (np.dot(rot,Xcentroid.T))
        return (trans,rot)
        # xid = np.array(xlist - Xcentroid)
        # yid = np.array(ylist - Ycentroid)
        # # print(xlist)
        # self.W = yid.dot(xid.T)
        # print(xlist)
        # for i in range(0,len(corresponds)):
        # #     # N += self.X[i].reshape(dim,1)@self.Y.reshape(1,dim)
        # #     # self.dev += 
        #     xlist.append(xid[i] - Xcentroid[i])
        #     ylist = yid[i] - Ycentroid[i]
            # self.dev.append(((xid[i] - Xcentroid),(yid[i] - Ycentroid))) #deviation from the centroid
            # # print(len(self.dev[1]))
            # self.W += (self.dev[i][0].dot(self.dev[i][1].T)) #calculate the covariance
        # N = self.X.T @ self.Y
        # self.W = (ylist.dot(xlist.T))
        # U,S,vt = np.linalg.svd(self.W)
        # rot = U.dot(vt.T)
        # trans = Ycentroid - (rot @ Xcentroid)
        # return (trans,rot)

    def compute(self,t,R,dmax,num_ICP_iters):
        tr = t
        ro = R
        # P_values = [np.transpose(self.X.copy())]
        # P_copy = np.transpose(self.X.copy())
        for i in range(num_ICP_iters):
            print(i)
            cor = self.EstimateCorrespondences(tr,ro,dmax)
            tr,ro = self.ComputeOptimalRigidRegistration(cor)
            icx = np.dot(ro,self.X.T) + tr
            icx = icx.T
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            ax.scatter(self.Y[0:,0], self.Y[0:,1], self.Y[0:,2], c='r', marker='o',s=0.5)
            ax.scatter(icx[0:,0], icx[0:,1], icx[0:,2], c='b', marker='^',s=0.5)
            ax.set_xlabel('X Label')
            ax.set_ylabel('Y Label')
            ax.set_zlabel('Z Label')
            filename = str(i)+'plot.png'
            fig.savefig(filename)
            plt.close(fig)
            # self.plot2gif((self.Y),(icx),i)
            # P_copy = ro.dot(P_copy)
            # P_values.append(P_copy)
        return tr,ro



        
if __name__ == "__main__":
    x = np.loadtxt("C:/Users/yashm/Downloads/pclX.txt", dtype=float)
    xpcd = np.matrix(x)
    y = np.loadtxt("C:/Users/yashm/Downloads/pclY.txt", dtype=float)
    ypcd = np.matrix(y) 
    icp = IterativeClosedPoint(xpcd,ypcd)
    R = np.matrix([[1,0,0],[0,1,0],[0,0,1]])
    t = np.matrix([[0],[0],[0]])
    # start = time.time()
    trans,rot = icp.compute(t,R,0.25,30)
    icx = np.dot(rot,xpcd.T) + trans
    # icx = icx.T
    # icy = rot@xpcd.T + trans
    rmse = np.sqrt(np.square(np.subtract(ypcd,icx.T)).mean())

    print(rmse)
    # icy = icy.T
    print(icx.shape)
    print(xpcd.shape)
    # print(pts.T.shape)
    # print("The ICP took %s seconds to compute" % (time.time()-start))
    # print(trans)
    # trans,rot = icp.ComputeOptimalRigidRegistration(icp.EstimateCorrespondences(t,R,0.25))
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(icx[0], icx[1], icx[2], c='r',s=0.5)
    ax.scatter(ypcd[:,0], ypcd[:,1], ypcd[:,2], c='b', s=0.5)
    # ax.scatter(xpcd[:,0], xpcd[:,1], xpcd[:,2], c='g',s=0.5)
    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')

    plt.show()
    print("Translation Matrix: ",trans.T)
    print("Rotation Matrix: ",rot)
    # print(icp.EstimateCorrespondences(t,R,10))

