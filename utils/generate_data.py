import numpy as np

class PointCloud():
    def __init__(self, n_samples=1000, n_points='random', noise=0.05):
        '''
        Generate point cloud data sampled from sphere, torus, Mobius strip and Klein bottle.
            
            Args :
                n_samples (int) : The number of samples, i.e. the size of each dataset.
                n_points (int or str) : The number of points in each point cloud. 
                                        If n_points = 'random', assign n_points between 1000 and 2000 randomly.
                noise (float) : The standard deviation of noise to include each point cloud.
        '''
        self.n_samples = n_samples
        self.n_points = n_points
        self.noise = noise

    def sphere(self, r=1.0):        
        '''
        Generate point cloud examples from sphere.
            Arg :
                r (float) : the radius of sphere.
            Return :
                data (list) : The list containing the `n_samples` number of point clouds \nwhich consist of `n_points` points.
        '''
        data = []

        for _ in range(self.n_samples):
            if self.n_points == 'random':
                self.n_points = np.random.randint(1000,2000,1)[0]
            
            sample = np.zeros([self.n_points, 3])
            
            # domain of parameters
            t = np.random.uniform(0, 1*np.pi, self.n_points)
            s = np.random.uniform(0, 2*np.pi, self.n_points)
        
            # parametric equation of sphere
            sample[:, 0] = r*np.cos(s)*np.sin(t)   # x
            sample[:, 1] = r*np.sin(s)*np.sin(t)   # y
            sample[:, 2] = r*np.cos(t)             # z

            # random noise
            noise = np.random.normal(0, self.noise, size=sample.shape)
            data.append(sample + noise)
        return data

    def torus(self, R=2, r=1):
        '''
        Generate point cloud examples from torus.
            Arg :
                R (float) : the bigger radius of torus.
                r (float) : the smaller radius of torus. (R>r)
            Return :
                data (list) : The list containing the `n_samples` number of point clouds \nwhich consist of `n_points` points.
        '''
        data = []

        for _ in range(self.n_samples):
            if self.n_points == 'random':
                self.n_points = np.random.randint(1000,2000,1)[0]
            
            sample = np.zeros([self.n_points, 3])
            
            # domain of parameters
            t = np.random.uniform(0, 2*np.pi, self.n_points)
            s = np.random.uniform(0, 2*np.pi, self.n_points)
        
            # parametric equation of sphere
            sample[:, 0] = (r*np.cos(s)+R)*np.cos(t)   # x
            sample[:, 1] = (r*np.cos(s)+R)*np.sin(t)   # y
            sample[:, 2] = r*np.sin(s)             # z

            # random noise
            noise = np.random.normal(0, self.noise, size=sample.shape)
            data.append(sample + noise)
        return data

    def mobius(self):
        '''
        Generate point cloud examples from Mobius band.
            Arg :
                
            Return :
                data (list) : The list containing the `n_samples` number of point clouds \nwhich consist of `n_points` points.
        '''
        data = []

        for _ in range(self.n_samples):
            if self.n_points == 'random':
                self.n_points = np.random.randint(1000,2000,1)[0]
            
            sample = np.zeros([self.n_points, 3])
            
            # domain of parameters
            t = np.random.uniform(-0.5, 0.5, self.n_points)
            s = np.random.uniform(0.0, 2*np.pi, self.n_points)
        
            # parametric equation of sphere
            sample[:, 0] = (1-t*np.sin(s/2))*np.cos(s)   # x
            sample[:, 1] = (1-t*np.sin(s/2))*np.sin(s)   # y
            sample[:, 2] = t*np.cos(s/2)             # z

            # random noise
            noise = np.random.normal(0, self.noise, size=sample.shape)
            data.append(sample + noise)
        return data

    def klein_bottle(self):
        '''
        Generate point cloud examples from Klein bottle.
            Arg :
                
            Return :
                data (list) : The list containing the `n_samples` number of point clouds \nwhich consist of `n_points` points.
        '''
        data = []

        for _ in range(self.n_samples):
            if self.n_points == 'random':
                self.n_points = np.random.randint(1000,2000,1)[0]
            
            sample = np.zeros([self.n_points, 3])
            
            # domain of parameters
            u = np.random.uniform(0.0, np.pi, self.n_points)
            v = np.random.uniform(0.0, 2*np.pi, self.n_points)
        
            # parametric equation of sphere
            sample[:, 0] = (-(2/15) * np.cos(u) * 
                                (3 * np.cos(v) 
                                - 30* np.sin(u) 
                                + 90 * (np.cos(u) ** 4) * np.sin(u)
                                - 60 * (np.cos(u) ** 6) * np.sin(u)
                                + 5 * np.cos(u) * np.cos(v) * np.sin(u)))  # x
            sample[:, 1] = (-(1/15) * np.sin(u) * 
                                (3 * np.cos(v) 
                                - 3* (np.cos(u) **2) * np.cos(v) 
                                - 48 * (np.cos(u) ** 4) * np.cos(v)
                                + 48 * (np.cos(u) ** 6) * np.cos(v)
                                - 60 * np.sin(u)
                                + 5 * np.cos(u) * np.cos(v) * np.sin(u)
                                - 5 * (np.cos(u) ** 3) * np.cos(v) * np.sin(u)
                                - 80 * (np.cos(u) ** 5) * np.cos(v) * np.sin(u)
                                + 80 * (np.cos(u) ** 7) * np.cos(v) * np.sin(u)
                                ))   # y
            sample[:, 2] = (2/15) * (3 + 5 * np.cos(u) * np.sin(u)) * np.sin(v)         # z

            # random noise
            noise = np.random.normal(0, self.noise, size=sample.shape)
            data.append(sample + noise)
        return data