from sklearn.decomposition import PCA
from matplotlib import pyplot
import matplotlib.pyplot as plt


def plot_pca_components(src, iteration, dataset_name):
    batch_size, _ = src.size()
    src = src.clone().data.cpu().numpy()
    for b_index in range(batch_size):
        pca = PCA(n_components=3)
        projected = pca.fit_transform( np.reshape(src[b_index], [-1, 32] )  )
        fig = pyplot.figure()
        ax = Axes3D(fig)
        ax.scatter(projected[:, 0], projected[:, 1], projected[:, 2], marker='+')
        pyplot.show()
        plt.savefig('./plots/pca_'+str(iteration)+'.png')
        plt.close()