# Somya Dahiaya


import numpy as np
import matplotlib.pyplot as plt


def galton(height,no):
    distribution=[]
    # print(distribution)
    i=0
    for i in range(0,100000):
        a= np.random.randint(0,2,height)
        position=np.where(a==0,-1,1)
        # print(position)
        bin=np.sum(position)

        distribution.append(bin)
        i=i+1

    # print(distribution)

    plt.hist(distribution,bins=height+1,density=True, label=f'height={height}')
    plt.title(f'Galton board for h={height}')
    plt.xlabel('pocket')
    plt.ylabel('normalised count')
    plt.legend()
    plt.savefig(f"2d{no}.png")
    plt.close()


if __name__ == "__main__":

    heights=(10,50,100)

    i=1

    for h in heights:
        galton(h,i)
        i=i+1
