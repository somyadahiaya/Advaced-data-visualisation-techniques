# Somya Dahiaya

import numpy as np
from scipy.stats import norm
import matplotlib.pyplot as plt
import math

uni=np.random.uniform(low=0.0,high=1.0,size=100000)


def sample(loc,scale):

    normal=norm.ppf(uni,loc,scale)

    return normal

    
example=[(0, 0.2), (0, 1.0), (0, 5.0), (-2, 0.5)]    


for lo,sc in example:
    final=sample(lo,np.sqrt(sc))

    plt.hist(final, bins=1000,density=True,alpha=0.5,label=f'μ={lo}, σ^2={sc}')
   

    plt.title(f'Samples from Gaussian Distribution')
    plt.xlabel('x')
    plt.ylabel('p(x)')
    plt.legend()
    plt.show()
    plt.savefig("2c.png")

