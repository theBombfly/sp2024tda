import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import MDS
distances = np.load('distance_matrix.npy')
print(distances)
CLASSES = ['Mild_Demented',
           'Moderate_Demented',
           'Non_Demented',
           'Very_Mild_Demented']
mds = MDS(n_components =3, metric =True, dissimilarity = 'precomputed')
Xmds = mds.fit_transform(distances)
plt.rcParams.update(plt.rcParamsDefault)
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(Xmds[0:50,0],Xmds[0:50,1],Xmds[0:50,2])
ax.scatter(Xmds[50:100,0],Xmds[50:100,1],Xmds[50:100,2])
ax.scatter(Xmds[100:150,0],Xmds[100:150,1],Xmds[100:150,2])
ax.scatter(Xmds[150:201,0],Xmds[150:201,1],Xmds[150:201,2])
plt.legend(CLASSES)
plt.show()
