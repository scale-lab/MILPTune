import matplotlib.pyplot as plt
from sklearn.manifold import TSNE


def plot_tsne(X, y, file_name, colormap=plt.cm.Blues):
    plt.figure(figsize=(8, 6))

    # clean the figure
    plt.clf()

    tsne = TSNE(learning_rate="auto", init="pca")
    # tsne = TSNE(learning_rate="auto", init="pca", perplexity=5)
    # tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=colormap)

    plt.xticks(())
    plt.yticks(())
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(file_name)
