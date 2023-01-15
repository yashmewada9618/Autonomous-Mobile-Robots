import numpy as np
import scipy.stats
from scipy.linalg import expm
from matplotlib import pyplot as plt
class Particle_Filter:
    def __init__(self,num_particles) -> None:
        self.sigma_r = 0.05 # std of right wheel velocitys
        self.sigma_l = 0.05 # std of left wheel velocitys
        self.sigma_p = 0.1 # # std of positions from GPS
        self.num_particles = num_particles #number of particles
        # self.weights = np.ones([self.num_particles,2])/self.num_particles #initial weights of all the particles
        self.x_init = 1.*np.identity(3)
        self.new_particles = np.zeros((self.num_particles,2))
    
    def motion_model(self,t2,u,r,w):
        """
        Args:
            t2: The time at which we want to find the position of bot.
            u:  The commanded wheel speeds at titme t2. 
            r:  radius of wheels.
            w:  Width of track
        returns:
            motion_model: The final transformation matrix based on the input command at desired time.
        """
        manifold_map = np.matrix([[0.,-r*(u[0] - u[1])/w,r*(u[0] + u[1])/2],
                                  [r*(u[0] - u[1])/w ,0.,0.],
                                  [0.,0.,0.]])
        motion_model = self.x_init@expm(t2*manifold_map)
        return motion_model

    def ParticleFilterPropagate(self,t1,particles,u,t2,r,w):
        """
        Args:
            t1: Current time.
            particles: Initial particle set which describes robot belief. This is similar to p(x1).
            u: commanded wheel speeds (right_wheel_speed,lleft_wheel_speed)
            t2: Time at which you want to propagate the robot's belief provided the wheel speeds of time t2.
            r: radius of wheels.
            w: Width of track
        returns:
            pose_true: True position of the robot based on the commanded wheel speeds without noise.
            particles: belief the robot thinks it is based on the noise distribution int the wheel speeds
        """
        flag = True
        if flag == True:
            init_particles =np.zeros((self.num_particles,2))
            ud1 = np.random.normal(loc=0,scale = self.sigma_r,size = self.num_particles)
            ud2 = np.random.normal(loc=0,scale = self.sigma_l,size = self.num_particles)
            init_particles[:,0] += ud1
            init_particles[:,1] += ud2
            ud1 = 0
            ud2 = 0
            flag = False
        
        pose_true = self.motion_model(t2,u,r,w)
        pose_true = np.array((pose_true.item((0,2)),pose_true.item((1,2)))) # extract the X and Y coordinates from the transformation
        my_list =np.zeros((self.num_particles,2))
        ud1 = u[0] + np.random.normal(loc=0,scale = self.sigma_r,size = self.num_particles)
        ud2 = u[1] + np.random.normal(loc=0,scale = self.sigma_l,size = self.num_particles)
        my_list[:,0] += ud1
        my_list[:,1] += ud2

        i = 0
        for ir,jl in my_list:
            pose = self.motion_model(t2,(ir,jl),r,w)
            particles[i,0] = pose.item((0,2))
            particles[i,1] = pose.item((1,2))
            i += 1
        
        return pose_true,particles,init_particles

    def ParticleFilterUpdate(self,particles,z,weights):
        """
        Args:
            particles: Initial particle set which describes robot belief. This is similar to p(x1).
            z: noisy measurement from GPS at time t2.
            sigma_p: given variance of the noisy GPS measurement.
        retunrs:
            particles: The particle set based on the posterior belief given the measurement from sensor.
        """
        weights.fill(1.) # initialise weights with 1
        for i in range(len(z)):
            weights *= scipy.stats.norm(particles[:,i],self.sigma_p).pdf(z[i]) # find the weights of each particle based on their distribution and posterior belief.
        print(weights.shape)
        weights /= sum(weights) #normalise the weights i.e p(z).

        positions = (np.arange(N) + np.random.random()) / N # sample the random position with size equal to that of weights
        indexes = np.zeros(N, 'i') #initialise indexes as int datatype.
        cumulative_sum = np.cumsum(weights) # find the cummulative sum of all weights.
        i, j = 0, 0
        while i < self.num_particles and j < self.num_particles:
            if positions[i] < cumulative_sum[j]: #if the sampled position index is less than cummulative sum of that index, then add that index.
                indexes[i] = j
                i += 1
            else:
                j += 1

        particles[:] = particles[indexes] # replace that particles with sampled indexes
        return particles

if __name__ == '__main__':
    N = 1000
    px = np.zeros((N, 2))  # Particle store
    pxp = np.zeros((N,2))
    # pw = np.ones((N,1)) / N
    px = np.empty((N, 2))
    px[:, 0] = np.random.uniform(0, 50, size=N)
    px[:, 1] = np.random.uniform(0, 50, size=N)
    pw = np.array([1.0]*N)
    particlefilter = Particle_Filter(N)
    # px = particlefilter.create_gaussian_particles(0,(5, 5, np.pi/4), N)
    phi_l = 1.5
    phi_r = 2.0
    w = 0.5
    r = 0.25
    u = [phi_r,phi_l]

    # _,x,init_particles = particlefilter.ParticleFilterPropagate(0,px,u,0,r,w)
    x,pose,_ = particlefilter.ParticleFilterPropagate(0,px,u,10,r,w)

    
    # mean = np.average(pose,axis=0)
    # cov = np.average((pose-mean)**2,axis=0)
    # print("Emperical Mean:",mean)
    # print("Covariance:",cov)
    # print(mean)
    samp_t = np.arange(5,25,5)
    print(samp_t)
    fig = plt.figure()
    ax = fig.add_subplot()
    # ax.scatter(pose[:,0], pose[:,1],label = 'propagate')
    # ax.scatter(init_particles[:,0], init_particles[:,1], c='r',s=0.5)
    # ax.scatter(x[0], x[1], c='black',label = 'true pose')
    # ax.scatter(mean[0], mean[1], c='r',label = 'mean')
    zs = [(1.6561,1.2847),(1.0505,3.1059),(-0.9875,3.2118),(-1.6450,1.1978)]
    ax.legend()
    # new_particles = particlefilter.ParticleFilterUpdate(px,zs[1],pw)
    j = 0
    num_iters = 10
    # for i in samp_t:
    #     x_true,pose,_ = particlefilter.ParticleFilterPropagate(0,px,u,i,r,w)
    #     ax.scatter(pose[:,0], pose[:,1],label = 'propagate'+ str(i),s=0.5)
    for i,f in zip(samp_t,zs):
        x,pose,init_particles = particlefilter.ParticleFilterPropagate(0,px,u,20,r,w) #propagate step
        # new_particles = particlefilter.ParticleFilterUpdate(pose,f,pw) #measurement update step
        
        mean = np.mean(pose,axis=0)
        std = np.stack((pose[:,0],pose[:,1]),axis = 0)
        cov = np.cov(std)
        print("Emperical Mean:",mean,"for t:",i)
        print("Covariance:",cov,"for t:",i)
        ax.scatter(pose[:,0], pose[:,1],label = 'propagate'+ str(i),marker='o')
        # ax.scatter(new_particles[:,0], new_particles[:,1],label = 'update at t:' + str(i))
        # ax.scatter(x[0], x[1],c='black',label='Robot Pos')
        # ax.scatter(0, 0,c='black')
        # ax.scatter(f[0], f[1],c='r',label='GPS')
        ax.legend()
        # pose = px
        # pose_particles = px
        # pose_particles = np.zeros((N, 2))
        # filename = 'pf-' + str(i) +'.png'
        # fig.savefig(filename)
        # plt.close(fig)
        # mean = np.average(pose,axis=0)
        # cov = np.average((pose-mean)**2,axis=0)
        # print("Emperical Mean:",mean,"for t:",i)
        # print("Covariance:",cov,"for t:",i)
    # x,pose,_ = particlefilter.ParticleFilterPropagate(0,px,u,20,r,w)
    # pose_particles = particlefilter.ParticleFilterUpdate(pxp,zs[3],pw)
    # ax.scatter(pose_particles[:,0], pose_particles[:,1], c='black',s=0.5)
    # ax.scatter(pose[:,0], pose[:,1],s=0.5)
    # pose_particles = px
    # for i in samp_t:
    #     # for i in range(num_iters):
    #     # print(s)
    #     x_true,pose,_ = particlefilter.ParticleFilterPropagate(0,px,u,i,r,w)
    # #     # print(x)
    # #     pose_particles = particlefilter.ParticleFilterUpdate(pose,s,pw)
    # #     # print(pose_particles.shape)
    # #     # emperical_mean[j,0] = np.mean(x[0])
    # #     # emperical_mean[j,1] = np.mean(x[1])
    # #     # cov[j,0] = np.cov(x[0])
    # #     # cov[j,1] = np.cov(x[1])
    # #     j += 1
    # #     ax.scatter(pose_particles[:,0], pose_particles[:,1], c='black')
    #     ax.scatter(pose[:,0], pose[:,1],s=0.5)
    #     # ax.scatter(px[:,0], px[:,1], c='g',s=0.5)
    #     print(x)
    #     ax.scatter(s[0], s[1], c='r')
    #     ax.scatter(x_true[0], x_true[1], c='black')
    # #     px = np.zeros((N, 2))
    #     # pose = px
    #     # pose_particles = px
    #     # pose_particles = np.zeros((N, 2))
    #     # filename = 'plot-' + str(i) +'.png'
    #     # fig.savefig(filename)
    #     # plt.close(fig)
    # print(emperical_mean,"Emperical Means for all timesteps")
    # print(cov   ,"Covariance for all timesteps")
    # ax.scatter(pose_5[:,0], pose_5[:,1], c='g',s=0.5)
    # ax.scatter(pose_10[:,0], pose_10[:,1], c='r',s=0.5)
    # ax.scatter(pose_15[:,0], pose_15[:,1], c='b',s=0.5)
    # ax.scatter(pose_20[:,0], pose_20[:,1], c='black',s=0.5)
    # particlefilter.motion_model()
    # print(dr)
    # print(prt)
    # print(ud)
    
    # print(px.shape)
    # x_t = dr.item((0,2))
    # y_t = dr.item((1,2))
    # # x_et = ud.item((0,2))
    # # y_et = ud.item((1,2))
    # ax.scatter(x_t, y_t, c='r')
    # ax.scatter(x_et, y_et, c='r')
    # ax.scatter(prt[:,0], prt[:,1], c='b',s = 0.5)
    # plt.xlim([-5,10])
    # plt.ylim([-5,10])
    plt.show()
    # print(my_part.shape)
    # for p in my_part:
    #     plt.plot(p[0],p[1],color = 'white')
    # t0 = 0
    # t1 = 10
    # phi_l = 1.5
    # phi_r = 2
    # w = 0.5
    # r = 0.25
    # u = [phi_l,phi_r]
    # propogate = particlefilter.ParticleFilterPropagate(t0,my_part,u,t1,(r,w))
    # for p in propogate:
    #     plt.plot(p[0],p[1],color = 'white')
    # plt.show()
    # print(propogate)
    # print("hellow")