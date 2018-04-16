import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
np.set_printoptions(threshold=np.nan)




'''
Grid2 = np.concatenate((Grid, [Grid[0,:]]), axis=0)
Grid2 = np.concatenate((Grid2, Grid2[:,0].reshape(col+1,1)), axis=1)
def EnSite(i,j):
    return Grid[i,j]*(Grid2[i,j+1]+Grid2[i+1,j])
'''

def EnSite(i,j, Grid):
    if i != N-1 and j != N-1:
        return -J*Grid[i,j]*(Grid[i,j+1]+Grid[i+1,j])
    elif i != N-1 and j == N-1:
        return -J*Grid[i,j]*(Grid[i,0]+Grid[i+1,j])    
    elif i == N-1 and j != N-1:
        return -J*Grid[i,j]*(Grid[i,j+1]+Grid[0,j]) 
    else:
        return -J*Grid[i,j]*(Grid[i,0]+Grid[0,j])         




def FullEnSite(i,j, Grid):
    if i == 0 and j == 0:
        return -J*Grid[0,0]*(Grid[0,1] + Grid[1,0] + Grid[0,-1] + Grid[-1,0])
    
    elif i == N-1 and j == N-1:
        return -J*Grid[i,j]*(Grid[i,j-1] + Grid[i-1,j] + Grid[0,j] + Grid[i,0])   
    
    elif i == 0 and j == N-1:
        return -J*Grid[0,j]*(Grid[0,j-1] + Grid[1,j] + Grid[0,0] + Grid[-1,j])  
    
    elif i == N-1 and j == 0:
        return -J*Grid[i,0]*(Grid[i,1] + Grid[i-1,0] + Grid[i,-1] + Grid[0,0])  
    
    elif i == 0:
        return -J*Grid[0,j]*(Grid[0,j+1] + Grid[1,j] +  + Grid[0,j-1] + Grid[-1,j]) 
       
    elif i == N-1:
        return -J*Grid[i,j]*(Grid[i,j+1] + Grid[i-1,j] +  + Grid[0,j] + Grid[i,j-1]) 
    
    elif j == 0:
        return -J*Grid[i,0]*(Grid[i,1] + Grid[i-1,0] + Grid[i+1,0] + Grid[i,-1])  
    
    elif j == N-1:
        return -J*Grid[i,j]*(Grid[i,j-1] + Grid[i-1,j] + Grid[i+1,j] + Grid[i,0])  
          
    else:
        return -J*Grid[i,j]*(Grid[i,j+1] + Grid[i,j-1] + Grid[i+1,j] + Grid[i-1,j])         





def Energy(Grid):
    En = 0
    for i in range(N):
        for j in range(N):
            En = En - EnSite(i,j, Grid)
     
    return En




def MMC(Grid, T, N):

    for i in range(1000000):
    
        a = np.random.randint(N)
        b = np.random.randint(N)
    
        
        new_Grid = np.copy(Grid)
        new_Grid[a,b] = np.negative(new_Grid[a,b])
    
        
        if FullEnSite(a,b, new_Grid) <= FullEnSite(a,b,Grid):
            Grid[a,b] = np.negative(Grid[a,b])
            
        elif np.random.rand() < np.exp(-1/T*(FullEnSite(a,b,new_Grid) - FullEnSite(a,b,Grid))):
            Grid[a,b] = np.negative(Grid[a,b])
            
        else:
            pass

    return Grid
    


def MakeX(n_samples, N):
    X = np.zeros([1,N**2], dtype=int)
    
    for T in np.linspace(1.6, 2.9, num=14):

        for samples in range(n_samples):
            
            Grid = np.random.randint(2, size=(N,N))
            Grid[Grid == 0] = -1
            #print MMC(Grid, T, N)
            X = np.append(X, [np.reshape(MMC(Grid, T, N),N**2)], axis = 0)

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


J=1
N=24
n_samples = 20


X = MakeX(n_samples, N)


pca = PCA(n_components=2)
pca.fit(X)
X_new = pca.transform(X)

x = X_new[:,0]
y = X_new[:,1]


print(pca.explained_variance_ratio_)  
print(pca.singular_values_)  
print pca.components_.shape
print X_new.shape

plot(x,y)








