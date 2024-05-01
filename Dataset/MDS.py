import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.manifold import MDS
dic = np.load('datadumper.npz')
Xt = dic['X_test']
Xt = np.array([img.flatten() for img in tqdm(Xt)])
y = np.argmax(dic['y_test'], axis=1)
CLASSES = ['Mild_Demented',
           'Moderate_Demented',
           'Non_Demented',
           'Very_Mild_Demented']
mds = MDS(n_components =2)
Xmds = mds.fit_transform(Xt)
plt.rcParams.update(plt.rcParamsDefault)
cmap=plt.colormaps['rainbow']
#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
#ax.scatter(Xmds[:,0], Xmds[:,1], Xmds[:,2], c = y[:100], cmap=cmap)
plt.scatter(Xmds[:,0], Xmds[:,1], c = y, cmap=cmap)

plt.show()
