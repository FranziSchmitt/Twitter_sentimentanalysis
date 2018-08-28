import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import silhouette_samples


"""Pre-Clustering visualizations"""
def silhouette_plot(X, y):

    silhouette_vals_km = silhouette_samples(X, y, metric='euclidean')

    cluster_labels = np.unique(y)
    n_clusters = cluster_labels.shape[0]
    y_ax_upper, y_ax_lower = 0, 0
    yticks = []

    for i, c in enumerate(cluster_labels):
        c_silhouette_values = silhouette_vals_km[y == c]
        c_silhouette_values.sort()
        y_ax_upper += len(c_silhouette_values)
        color = cm.get_cmap('hot')(float(i) / n_clusters)
        plt.barh(range(y_ax_lower, y_ax_upper),
            c_silhouette_values,
            height=1.0,
            edgecolor='none',
            color=color)
        yticks.append((y_ax_lower + y_ax_upper) / 2.)
        y_ax_lower += len(c_silhouette_values)

    silhouette_avg = np.mean(silhouette_vals_km)
    plt.axvline(silhouette_avg, color='grey', linestyle='--')
    plt.yticks(yticks, cluster_labels + 1)
    plt.ylabel('Cluster')
    plt.xlabel('Silhouette-Coefficient')
    
    
def silhouette_value(X, y):
    
    silhouette_vals_km = silhouette_samples(X, y, metric='euclidean')
    silhouette_avg = np.mean(silhouette_vals_km)
    return silhouette_avg

def elbow_values(X, model, startend, **kwargs):
    values = []

    if len(startend) == 2:
        x = [e for e in range(startend[0], startend[1])]
    else:
        x = startend
    for i in x:
        print(i)
        model_out = model(n_clusters=i, **kwargs)
        y = model_out.fit_predict(X)
        values.append(silhouette_value(X, y))
    
    return values


def elbow_plot(X, model, startend, **kwargs):
    if len(startend) == 2:
        x = [e for e in range(startend[0], startend[1])]
    else:
        x = startend
    y = elbow_values(X, model, startend, **kwargs)
    
    f, ax = plt.subplots(1, 1)
    ax.plot(x, y, '.-')



"""Clustering visualizations"""
def tsne_plot(X, y, title, no_samples=None, savefig=False):
    cmi = 'viridis'
    X1 = X[:no_samples, 0]
    X2 = X[:no_samples, 1]
    c = y[:no_samples]
    
    fig = plt.figure(figsize=(15, 7.5))
    plt.scatter(X1, X2, c=c, s=50, cmap=cmi, alpha=0.7)
    plt.title(title)
    
    if savefig:
        fig.savefig('../Figures/CLST_'+title+'.png', bbox_inches='tight')


def threed_subplots(data, y, title, no_samples=None, savefig=False):
    svd = TruncatedSVD(n_components=3)
    X = svd.fit_transform(data)
    print(svd.explained_variance_ratio_)
    xs = X[:no_samples, 0]
    ys = X[:no_samples, 1]
    zs = X[:no_samples, 2]
    cmi = 'viridis'
    
    fig = plt.figure()
    fig, ax = plt.subplots(1, 3, figsize=(15, 7.5))
    ax[0].scatter(x=xs, y=ys, c=y, s=50, cmap=cmi, alpha=1)
    ax[1].scatter(x=xs, y=zs, c=y, s=50, cmap=cmi, alpha=1)
    ax[2].scatter(x=ys, y=zs, c=y, s=50, cmap=cmi, alpha=1)
    plt.title(title)
    
    if savefig:
        fig.savefig('../Figures/CLST_'+title+'.png', bbox_inches='tight')


def threed_plot(data, y, title, no_samples=None, savefig=False):
    svd = TruncatedSVD(n_components=3)
    X = svd.fit_transform(data)
    print(svd.explained_variance_ratio_)
    xs = X[:no_samples, 0]
    ys = X[:no_samples, 1]
    zs = X[:no_samples, 2]
    c = y[:no_samples]
    cmi = 'viridis'
    
    fig = plt.figure(figsize=(20, 7.5))
    ax = fig.add_subplot(111, projection='3d')   
    ax.scatter(xs, ys, zs, c=c, s=50, cmap=cmi)
    plt.title(title)
    
    if savefig:
        fig.savefig('../Figures/CLST_'+title+'.png', bbox_inches='tight')


def twod_plot(data, y, title, no_samples=None, savefig=False):
    svd = TruncatedSVD(n_components=2)
    X = svd.fit_transform(data)
    print(svd.explained_variance_ratio_)
    xs = X[:no_samples, 0]
    ys = X[:no_samples, 1]
    cmi = 'viridis'
    
    fig = plt.figure(figsize=(15, 7.5))
    plt.scatter(x=xs, y=ys, c=y, s=50, cmap=cmi, alpha=1)
    plt.title(title)
    
    if savefig:
        fig.savefig('../Figures/CLST_'+title+'.png', bbox_inches='tight')