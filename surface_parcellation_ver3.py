"""

Script for surface-based parcellation. Based on the Frontiers paper and codes by
Thririon, B. (2014).

"""

import nibabel
import numpy as np
import time as time
import matplotlib.pyplot as plt
from scipy.sparse import coo_matrix
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabaz_score
from scipy import stats

#==============================================================================
# Loading data & compute z-score for each parameter (e.g. R1, T2*...)
#==============================================================================
parameters = ['R1','T2','thickness'] # which parameters to include

print 'loading surface and input data for parcellation...'
surf        = nibabel.freesurfer.io.read_geometry('D:/MR_data/HBM_myelin/analyzed/fsaverage/surf/rh.inflated')
nverts      = len(surf[0])
nfaces      = len(surf[1])
faces       = surf[1]
coords      = surf[0]

data = np.zeros((nverts,len(parameters)))
for j in range(0,len(parameters)):
    data_tmp = nibabel.freesurfer.io.read_morph_data('D:/MR_data/HBM_myelin/analyzed/fsaverage/parcellation/rh.%s' % parameters[j])
    data_tmp = stats.zscore(data_tmp)
    for i in range(0,nverts):
        data[i,j]=data_tmp[i]

#==============================================================================
# Some variables used for calculating the silhouette scores
#==============================================================================
n_iterations= 20
size_dist   = 200

print '1. finding number of nearest neighbors...'
st = time.time()
num_nbrs = [0] * nverts

for i in range(0, nfaces):
    num_nbrs[faces[i,0]] = num_nbrs[faces[i,0]]+1
    num_nbrs[faces[i,1]] = num_nbrs[faces[i,1]]+1
    num_nbrs[faces[i,2]] = num_nbrs[faces[i,2]]+1

max_num_nbrs = max(num_nbrs)
print '     Done. Elapsed time (sec): ', time.time() - st

print '2. finding nearest neighbors...'
st = time.time()
nbrs = np.zeros((nverts, max_num_nbrs))

for i in range(0, nfaces):
    for j in range(0, 3):
        vcur = faces[i,j]
        for k in range(0, 3):
            if j != k:
               vnbr = faces[i,k] 
               if vnbr in nbrs[vcur,:]:  
                   pass
               else:         
                   n_nbr = min(np.where(nbrs[vcur,:] == 0))
                   n_nbr = min(n_nbr)
                   nbrs[vcur,n_nbr] = vnbr   
            
print '     Done. Elapsed time (sec): ', time.time() - st

print '3. computing connectivity matrix...'
st = time.time()

num_non_zero_entries = np.sum(num_nbrs)
num_non_zero_entries = np.int64(num_non_zero_entries)
IndX = np.zeros(num_non_zero_entries,dtype=np.int64)
IndY = np.zeros(num_non_zero_entries,dtype=np.int64)

count = np.int(0);
for i in range(0,len(num_nbrs)):
    ThisSize = num_nbrs[i]
    IndX[range(0+count,ThisSize+count)] = i
    IndY[range(0+count,ThisSize+count)] = nbrs[i,range(0,ThisSize)]
    count  = count + ThisSize
    
connectivity = coo_matrix(([1]*num_non_zero_entries, (IndX, IndY)), shape = (nverts, nverts)) # Not displayed in variable explorer, but can be accessed. 

print '     Done. Elapsed time (sec): ', time.time() - st

#==============================================================================
# Generate vector of cluster numbers
#==============================================================================
#min_n_cluster=2
#max_n_cluster=50
#max_n_cluster2=100
#max_n_cluster3=1050
#range_n_clusters_tmp = np.arange(min_n_cluster,max_n_cluster,1).tolist()
#range_n_clusters_tmp2 = np.arange(max_n_cluster,max_n_cluster2,5).tolist()
#range_n_clusters_tmp3 = np.arange(max_n_cluster2,max_n_cluster3,50).tolist()
#range_n_clusters = range_n_clusters_tmp+range_n_clusters_tmp2+range_n_clusters_tmp3
#range_n_clusters.reverse()
#
#del min_n_cluster, max_n_cluster, max_n_cluster2, max_n_cluster3, range_n_clusters_tmp, range_n_clusters_tmp2, range_n_clusters_tmp3

range_n_clusters = [10]
silhouette_scores_avg = np.zeros((np.size(range_n_clusters)))
silhouette_scores_std = np.zeros((np.size(range_n_clusters)))
calinski_scores_avg = np.zeros((np.size(range_n_clusters)))
calinski_scores_std = np.zeros((np.size(range_n_clusters)))

print '4. compute structural hierarchical (Ward) clustering and silhouette scores...'

X = np.zeros((nverts,3))
X = data
silhouette_scores_index = 0
for c in range_n_clusters:
    print '   - %d clusters' % c
    st = time.time()  
    
    ward = AgglomerativeClustering(linkage='ward', n_clusters=c, connectivity=connectivity).fit(X)
    
#==============================================================================
# - Perform pre-whitening (scikit, decomposition.PCA preprocessing)
#    
# For pooling:
# - Truncation using percentiles per channel
# - NaN based on percentiles then mean
# - Different weights for features    
#==============================================================================
    
    
    exec('labels_%d_clusters' % c + " = ward.labels_") 
    labels = ward.labels_

    #ward_parcel_txt_string = 'D:/MR_data/HBM_myelin/analyzed/fsaverage/parcellation/rh_ward_%d_clusters.txt' % c
    #np.savetxt(ward_parcel_txt_string, labels % c, fmt='%1.1i')
    
    #==============================================================================
    # Generate borders using parcellation        
    #==============================================================================
    borders = np.zeros((nverts,1))
    c_nbrs = np.zeros((nverts,6))       
    for v in range(0, nverts):
        for v_nbr in range(0, num_nbrs[v]):
            c_nbrs[v,v_nbr] = labels[np.int(nbrs[v,v_nbr])]                
        if len(np.unique(c_nbrs[v])) > 1:
            borders[v] = labels[v]

    #borders_txt_string = 'D:/MR_data/HBM_myelin/analyzed/fsaverage/parcellation/rh_ward_%d_clusters_borders.txt' % c
    #np.savetxt(borders_txt_string, borders, fmt='%1.1i')                
        
    print '     computing silhouette scores between clusters after clustering...'
            
    #==============================================================================
    # Find neighbouring clusters
    #==============================================================================
    nbrs_clusters = np.zeros((nverts, max_num_nbrs))    
    for i in range(0, nverts):
        for j in range(0, 6):
            vcur = np.int(nbrs[i,j])
            nbrs_clusters[i,j] = labels[vcur]
    
    unique_nbrs_clusters = np.empty((c, c))
    unique_nbrs_clusters[:] = np.NAN
    for i in range(0, c):
        unique_nbrs_clusters[0:len(np.unique(nbrs_clusters[np.where(labels == i)])),i] = np.unique(nbrs_clusters[np.where(labels == i)])
    mask = np.all(np.isnan(unique_nbrs_clusters), axis=1)
    unique_nbrs_clusters = unique_nbrs_clusters[~mask]    
            
    #==============================================================================
    # Concatenate data+labels connected clusters
    #==============================================================================
    X_labeled = np.empty((nverts,4))
    X_labeled[:,0:3] = X[:,0:3]
    X_labeled[:,3] = labels
    
    #==============================================================================
    # Calculate silhoutte scores
    #==============================================================================
    avg_silhouette = np.zeros((c, 1))
    avg_calinski = np.zeros((c, 1))
    for n_cluster in range(0, c):
        print '     ...for cluster %d of %d...' % (n_cluster+1, c)
        n_connected_clusters = max(max(np.where(unique_nbrs_clusters[:,n_cluster] >= 0)))  
        input_silhouette = np.empty((1,1))
        silhouette_avg = [0] * n_iterations
        calinski_avg = [0] * n_iterations
        for j in range(0, n_iterations):        
            for i in range(0, n_connected_clusters + 1):
                if i == 0:
                    input_raw = X_labeled[np.where(labels == unique_nbrs_clusters[i,n_cluster]),:]
    
                    # Here we apply some random sampling for computational reasons to calculate the silhoutte score
                    if np.size(input_raw[0,:,0]) > size_dist:
                        mask = np.random.randint(len(input_raw[0,:,0]),size=size_dist)
                        input_silhouette = input_raw[0,mask,:]
                    else:
                        input_silhouette = input_raw[0,:,:]
                else:
                    input_raw = X_labeled[np.where(labels == unique_nbrs_clusters[i,n_cluster]),:]
                    # Again some random sampling for computational reasons to calculate the silhoutte score
                    if np.size(input_raw[0,:,0]) > size_dist:
                        mask = np.random.randint(len(input_raw[0,:,0]),size=size_dist)
                        input_raw_concat = input_raw[0,mask,:]
                        input_silhouette = np.vstack((input_silhouette, input_raw_concat))
                    else:
                        input_raw = input_raw[0,:,:]
                        input_silhouette = np.vstack((input_silhouette, input_raw))
                if n_connected_clusters == 0:
                    input_raw = X_labeled[np.where(labels == n_cluster),:]
                    input_raw = input_raw[0,:,:]
                    input_silhouette = np.vstack((input_silhouette, input_raw))
            X_input = np.empty((np.size(input_silhouette[:,0]),3))  
            X_input[:,0:3] = input_silhouette[:,0:3]
            silhouette_avg[j] = silhouette_score(X_input, input_silhouette[:,3], metric='euclidean')
            calinski_avg[j] = calinski_harabaz_score(X_input, input_silhouette[:,3])
        avg_silhouette[n_cluster] = np.min(silhouette_avg)        
        avg_calinski[n_cluster] = np.min(calinski_avg)
        
    print '     average....'    
    silhouette_scores_avg[silhouette_scores_index] = np.average(avg_silhouette)
    silhouette_scores_std[silhouette_scores_index] = np.std(avg_silhouette)/np.sqrt(len(avg_silhouette))
    calinski_scores_avg[silhouette_scores_index] = np.average(avg_calinski)
    calinski_scores_std[silhouette_scores_index] = np.std(avg_calinski)/np.sqrt(len(avg_calinski))
    silhouette_scores_index = silhouette_scores_index+1
    print '     Done and save. Elapsed time (sec): ', time.time() - st

#==============================================================================
# Plot silhouette scores across range of number of clusters
#==============================================================================
plt.errorbar(range_n_clusters, silhouette_scores_avg[:], silhouette_scores_std[:])
plt.axhline(y=0, color='k', ls='dashed')
plt.show()
   