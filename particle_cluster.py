# -*- coding: utf-8 -*-
"""
Created on Fri Aug 12 10:22:00 2016

@author: Lukas Gartmair
"""

import numpy as np
import matplotlib.pyplot as pl
import sklearn.cluster

shape = (200,200)

matrix = np.zeros(shape)

number_of_particles = 500

radius = 2

for i in range(number_of_particles):
    rnd_pos_x = np.random.randint(0,shape[0])
    rnd_pos_y = np.random.randint(0,shape[1])
    
    matrix[rnd_pos_x,rnd_pos_y] = 1
    #matrix[rnd_pos_x-radius:rnd_pos_x+radius,rnd_pos_y-radius:rnd_pos_y+radius] += 1
    
#pl.matshow(matrix*1000)


coords = np.where(matrix == 1)

Y = np.zeros((coords[0].size,2))

Y[:,0] = coords[0]
Y[:,1] = coords[1]


diagonal = np.sqrt(2)
straight = 1

epses = [diagonal]

for e in epses:
        
    db = sklearn.cluster.DBSCAN(eps=e, min_samples=2, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None).fit(Y)
    
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        
    # Black removed and is used for noise instead.
    unique_labels = set(labels)
    colors = pl.cm.Spectral(np.linspace(0, 1, len(unique_labels)))
    
    # number of different labels -> cluster entspircht unique labels - 1
    print(len(unique_labels)-1)    
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
            # Black used for noise.
            col = 'k'
    
        class_member_mask = (labels == k)
    
        xy = Y[class_member_mask & core_samples_mask]
    
        points = Y[class_member_mask]   
        
        # anzahl gesamtpunkte des clusters
        print(len(points))
        # anzahl der core punkte
        print(len(xy))
        
        # plot of the core samples
        pl.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=8)
        # plot der 'normalen' cluster punkte
        xy = Y[class_member_mask & ~core_samples_mask]
        pl.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=col,
                 markeredgecolor='k', markersize=8)

    pl.title('Estimated number of clusters: %d' % n_clusters_)
    pl.show()

        
