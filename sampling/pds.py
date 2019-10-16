"""
pds.py
============================================================================
Routines to perform sample-diversification for mini-batch stochastic gradient
descent/ascent.
"""
from sklearn.gaussian_process.kernels import RBF
from .sampler import sampler
import numpy as np




class PoissonDiskSampling(sampler):
    """
    Poisson disk sampling (PDS) is a data diversifcation method for stochastic 
    gradient descent for labelled categorical data.
    
    Point processes are actively sampled by using the categorical label to bias
    sampling.

    Arguments
    ---------
    X : np.ndarray, shape = [N,Ndim]
        Chosen representation of the input data .

    y : np.ndarray, type = int
        Integer categorical labels.

    method : str, default = "vanilla"
        Type of sampling bias to apply for Poisson process. Allowed values are :
        "vanilla", "easy", "dense", "anneal".

        +-----------+--------------------------------------------------------+
        | method    |                                                        |
        +===========+========================================================+
        | vanilla   | Ordinary PDS, where successive samples cannot be drawn |
        |           | from a Eucliden norm of np.std(X)*r0 from existing     |
        |           | points in a given sample                               |
        +-----------+--------------------------------------------------------+
        | easy      | Unlike vanilla PDS, here the repulsive radius is set   |
        |           | to 0 for points with a mingling index greater then 0 - |
        |           | only points surrounded exclusively by like-labelled    |
        |           | (easy to classify) points are repulsive. This biases   |
        |           | "easy" PDS to sample more difficult-to-classify points.|
        +-----------+--------------------------------------------------------+
        | dense     | For a mingling index group size of K, there are K      |
        |           | possible mingling index values for each point. When    |
        |           | proposing a new point, an arbitrary bias can be placed |
        |           | on the mingingling index of points by assigned         |
        |           | K coefficients to a categorical distribution that      |
        |           | determines the probabiliy that a point with a given    | 
        |           | mingling index is chosen at any time.                  |
        +-----------+--------------------------------------------------------+
        | anneal    | Sampling is biased by mingling index values as for     |
        |           | "dense" PDS, except for "anneal" PDS an iteration-     |
        |           | depedent categorical distribution can be specified. By |
        |           | default, the specified distribution initially biases   |
        |           | towards "easy" to classify points, and then moves      |
        |           | concentrates on "difficult" to classify points later.  |
        +-----------+--------------------------------------------------------+


    K : int, default = 4
        The mingling index group size. Mingling indices 

        .. math::
            :nowrap:       
 
            \[
            m_K(\\mathbf{x}_i) = \\frac{1}{K}\\sum_{j=1}^K \\delta^{-}(y_i,y_j)
            \]

        , where

        .. math::

            \\delta^{-}(y_i,y_j) &= 1, y_i \\neq y_j 
            
                                    &= 0, \\mathrm{otherwise}.

        and the indices :math:`j` iterate over the :math:`K` nearest neighbours
        to :math:`\mathbf{x}_i`, as determined by the Euclidean norm.


    r0 : float, default = 0.2
        A coefficient determining the size of the repulsive sphere surrounding
        points. The Eucliden norm np.std(X)*r0 is specified as the defailt 
        distance for repulsion to have effect. See self.method for a more
        detailed description.

    Examples
    --------
    ::
        
        from sampling.pds import PoissonDiskSampling

        # random samples in a 2-d space
        X = np.random.multivariate_normal([0,0],np.eye(2),size=50)

        # partition uper-right quadrant from rest of data
        y = [1 if all(_x>0) else 0 for _x in X]

        # instance of samper
        sampler = PoissonDiskSampling(X,y)
    """

    def __init__(self,X,y,method="vanilla",K=4,r0=0.2):
        self.set_X(X)
        self.set_y(y)       
 
        # mingling index group size
        self.K = K
 
        # threshold exclusion distance
        self.r = np.std(X)*r0
        
        # biasing scheme for drawing samples
        self._set_method(method)
        
        # default categorical distribution coefficients
        self._init_mingling_sampling_distribution()
        
        # pre-compute mingling indices
        self._calculate_mingling_indices()
        
        # for dart throwing
        self._calculate_distance_metrics()
        
        # important to keep track of samples drawn for annealing method
        self.iter = 0
        

    def set_annealing_schedule(self,pis):
        """
        Set the coefficients of the categorical distribution for sampling
        mingling values at each iteration of the sampler being called. If 
        more samples are drawn then iterations given here, the categorical
        distribution will be kept constant at the last values given here. 
        Annealing schedules for the categorical distribution can only be set 
        for when self.method=="anneal".

        Arguments
        ---------
        pis : np.ndarray, shape = [Niter][self.K]
            A 2-d array of categorical disitribution coefficients for mingling
            indices, given for each iteration of samples drawn from this class.

        Examples
        --------
        ::
    
            from sampling.pds import PoissonDiskSampling
            import numpy as np

            # uniformly sample from a 2-d space
            X = np.random.uniform(low=-2,high=2,size=(50,2))

            # parition by whether sin(X[:,0])>X[:,1]
            y = [1 if _x[1]>np.sin(_x[0]*np.pi*2) else 0 for _x in X]

            # number of mingling indices
            K = 5

            # specify anneal pds
            sampler = PoissonDiskSampling(X,y,method="anneal",K=K)
            
            # coefficients are normalized per iteration so no need to do this here
            coeffs = [(K-kk)*np.exp(-iter*(K-kk)/1000) for kk in range(K)] for iter in range(2000)]

            # set schedule for sampling mingling indices
            sampler.set_annealing_schedule(coefss)
    
        """
        if not isinstance(pis,(np.ndarray,list)): raise Exception(self.set_annealing_schedule.__doc__)
        elif not len(np.shape(pis))==2: raise Exception(self.set_annealing_schedule.__doc__)
        elif not np.shape(pis)[1]==self.K: raise Exception(self.set_annealing_schedule.__doc__)
        #elif self.method!="anneal": raise Exception(self.set_annealing_schedule.__doc__) 

        # [Niter][self.K] array of categorical distribution coefficients
        self.pis = pis
        if not isinstance(pis,np.ndarray):
            self.pis = np.asarray(self.pis)

        # ensure coefficients sum to 1 at each iteration
        self.pis /= np.tile(np.sum(self.pis,axis=1),(self.pis.shape[1],1)).T
    
    def sample(self,N):
        """
        Sample a subset of points of size N.

        Arguments
        ---------
        N : int
            The size of the subset. This must be less then the size of the 
            complete data set passed to this class when instantiated.

        Returns
        -------
        np.ndarray 
            The subset of points sampled for this point process.


        Examples
        --------
        ::
    
            from sampling.pds import PoissonDiskSampling
            import numpy as np
            import matplotlib.pyplot as plt

            # sample from bimodal 1-d distribution
            X = np.hstack((np.random.normal(-1,1,50),np.random.normal(1,1,50)))

            # partition about x=0
            y = [1 if _x>0 else 0 for _x in X]

            sampler = PoissonDiskSampling(X,y,method="dense")

            # draw samples disproportionately close to the boundary
            samples = np.asarray([sampler.sample(5) for ss in range(20)])

            # compare distributions
            _ = [plt.hist(_z,label=["data","samples"][ii]) for ii,_z in enumerate([y,samples.flatten()])]

            plt.legend()
            plt.show()
        """
        if N>self.X.shape[0]: raise Exception("Cannot sample more points than are present in the data")
        
        # list indices of selected points
        self.selected_points = []
        
        if self.method in ["dense","anneal"]:
            self._update_categorical_sampling_distribution()
        
        while len(self.selected_points)<N:
            # get index of proposed sample
            proposed_point = self._draw_sample()
        
            if self._accept_sample(proposed_point):
                self.selected_points.append(proposed_point)

        # for slicing
        self.selected_points = np.asarray(self.selected_points)

        # important for annealing
        self.iter += 1    
 
        # return x-coordinate and categorical labels
        return self.X[self.selected_points],self.y[self.selected_points]


    def _set_method(self,method):
        """
        Set type of bias applied when drawing samples

        Arguments
        ---------
        method : str, allowed values = ["vanilla","easy","dense","anneal"]
            The type of biasing method to use when drawing Poisson disk samples.
        """
        if not method in ["vanilla","easy","dense","anneal"]: 
            raise Exception("Unrecognised sampling method {}".format(method))
            
        self.method = method
        
    def _calculate_distance_metrics(self):
        """
        calculate distance between points as euclidean norm
        """
        self.distance = np.asarray([np.sqrt(np.sum(np.square(self.X-np.tile(_x,(self.X.shape[0],1))),axis=1)) \
                for _x in self.X])
       
                
    def _calculate_mingling_indices(self):
        """
        m(x_i) = 1/K sum_k delta(y_i,y_k) for K nearest neighbours to i
        
        Use Euclidean norm as metric for distance in X.
        """
        
        # compute kernel matrix for similarity measure between points
        self._calculate_kernel_matrix()
        
        # indices of nearest neighbours
        neighbours = [np.argsort(self.Cov[ii])[::-1][1:self.K+1] for ii in range(self.X.shape[0])]
        
        # anti-kronecker delta
        delta = lambda y1,y2 : 1.0 if y1!=y2 else 0.0 
        
        # mingingling indices - K discrete values
        self.m_index = np.asarray([np.sum([delta(self.y[ii],self.y[_id]) \
                for _id in neighbours[ii]]) for ii in range(self.X.shape[0])])
        
        # store indices of points by mingling index
        self.get_mingling_indices = {_k:np.where(self.m_index==_k)[0] for _k in range(self.K)}
        
        
    def _calculate_kernel_matrix(self):
        """
        Compute the kernel matrix over the input space
        """
        # use euclidean norm as distance metric
        kernel = RBF(1)
        
        # covariance matrix
        self.Cov = kernel(self.X,self.X)
        
    def _update_categorical_sampling_distribution(self):
        """
        If self.iter >= number if iterations provided in sampling schedule then
        use the last value present.
        """
        if self.iter > self.pis.shape[0]:
            self.pi = self.pis[self.pis.shape[0]-1]
        else:
            self.pi = self.pis[self.iter]


    def _accept_sample(self,proposed_idx):
        """
        check whether proposed point lies within exclusion disk of any other
        points.
        """
        if len(self.selected_points)==0:
            res = True
        else:
            # ordinary dart throwing
            OK_for_vanilla = lambda _x : not any([self.distance[_x][_id]<self.r for _id in self.selected_points])
        
            if self.method in ["vanilla"]:
                res = OK_for_vanilla(proposed_idx)
            elif self.method in ["easy","dense","anneal"]:
                if self.m_index[proposed_idx]>0:
                    # always accept if close-ish to boundary
                    res = True
                else:
                    # only points surrounded by like-categorized points repulse
                    res = OK_for_vanilla(proposed_idx)

        return res

    def _draw_sample(self):
        """
        return index of proposed sample - this can be a uniformly random or 
        biased choice
        """
        if self.method in ["vanilla","easy"]:
            # uniform sampling between accept/reject criteria
            
            res = np.random.choice(np.arange(self.X.shape[0]),1)[0]
        elif self.method in ["dense","anneal"]:
            # bias sampling by mingling value using coefficients of a sampling categorical distribution
            res = np.random.choice(self.get_mingling_indices[np.where(np.random.multinomial(1,self.pi))[0][0]],1)[0]
        else:
            raise NotImplementedError
        
        return res

    def _init_mingling_sampling_distribution(self):
        """
        Initialise default categorical distribution coefficients for mingling
        indices. For self.method == "anneal", this is a distribution that 
        exponentially decays the zero mingling index at the highest rate, 
        second lowest index at the next highest rate etc.
        """       
        if self.method == "anneal":
            coeffs = [[(self.K-kk)*np.exp(-iter*(self.K-kk)/1000) for \
                    kk in range(self.K)] for iter in range(2000)]
        else:
            coeffs = np.reshape(np.ones(self.K)/self.K,(1,-1))

        self.set_annealing_schedule(pis=coeffs)

 
