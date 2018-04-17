import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import time
np.set_printoptions(threshold=np.nan)
start = time.time()


def Idx(Grid,i,j,k):
    N = len(Grid)
    if i == N:
        i = 0
    if j == N:
        j = 0
    if k == N:
        k = 0
    return Grid[i,j,k,:]
        


def FullEnSite(Grid,i,j,k):
     return -np.dot(Grid[i,j,k,:], Idx(Grid,i+1,j,k)+Idx(Grid,i-1,j,k)
                      +Idx(Grid,i,j+1,k)+Idx(Grid,i,j-1,k)
                      +Idx(Grid,i,j,k+1)+Idx(Grid,i,j,k-1))                       



def MMC(Grid, T, N):

    for i in range(1000000):
    
        a = np.random.randint(N)
        b = np.random.randint(N)
        c = np.random.randint(N)    
        rand = np.random.randint(2)+1
        
        new_Grid = np.copy(Grid)
        new_Grid[a,b,c,:] = np.roll(new_Grid[a,b,c,:], rand)
    
        
        if FullEnSite(new_Grid,a,b,c) <= FullEnSite(Grid,a,b,c):
            Grid[a,b,c,:] = np.roll(new_Grid[a,b,c,:], rand)
            
        elif np.random.rand() < np.exp(-1/T*(FullEnSite(new_Grid,a,b,c) - FullEnSite(Grid,a,b,c))):
            Grid[a,b,c,:] = np.roll(new_Grid[a,b,c,:], rand)
            
        else:
            pass

    return Grid
  

def MakeGrid(N):
    Grid = np.random.randint(3, size=(N,N,N,3))+1 
    Grid[Grid[:,:,:,0]== 1,:] = np.array([1,0,0])
    Grid[Grid[:,:,:,0]== 2,:] = np.array([0,1,0])
    Grid[Grid[:,:,:,0]== 3,:] = np.array([0,0,1])
    return Grid



def MakeX(n_samples, N):
    X = np.zeros([1,3*N**3], dtype=int)
    
    for T in np.linspace(0.5, 1.8, num=14):

        for samples in range(n_samples):
            Grid = MakeGrid(N)
            X = np.append(X, [np.reshape(MMC(Grid, T, N),3*N**3)], axis = 0)

    X = np.delete(X, 0, axis=0)
    return X




def plot(x,y):

    a = np.linspace(1,14,14)
    b = np.ones(len(x)/14)
    color = np.outer(a,b).reshape(len(x))   
    color = np.round(color/14, decimals = 2)

    
    plt.scatter(x, y, c = color, cmap = 'rainbow')
    plt.show()

#--------------------------------------------------------------------------------


N=10
n_samples = 40
X = MakeX(n_samples, N)


pca = PCA(n_components=2)
pca.fit(X)
X_new = pca.transform(X)

x = X_new[:,0]
y = X_new[:,1]

print x
print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  
print pca.components_.shape
print X_new.shape

plot(x,y)







end = time.time()
print end-start