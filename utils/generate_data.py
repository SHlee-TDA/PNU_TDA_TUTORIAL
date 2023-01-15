import numpy as np

class PointCloud():
    def __init__(self, n_samples = 1000, n_points = 'random', manifolds=['sphere'],  noise = 0.05):
        self.n_samples = n_samples
        self.n_points = n_points
        self.noise = noise

    def sphere(self):        
        '''
        Generate point cloud examples.
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
            sample[:, 0] = np.cos(s)*np.sin(t)   # x
            sample[:, 1] = np.sin(s)*np.sin(t)   # y
            sample[:, 2] = np.cos(t)             # z

            # random noise
            noise = np.random.normal(0, self.noise, size=sample.shape)
            data.append(sample + noise)
        return data

    def torus(self):
        '''
        Generate point cloud examples.
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
            sample[:, 0] = (1*np.cos(s)+2)*np.cos(t)   # x
            sample[:, 1] = (1*np.cos(s)+2)*np.sin(t)   # y
            sample[:, 2] = 1*np.sin(s)             # z

            # random noise
            noise = np.random.normal(0, self.noise, size=sample.shape)
            data.append(sample + noise)
        return data

    def mobius(self):
        '''
        Generate point cloud examples.
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
        Generate point cloud examples.
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