from sklearn.decomposition import PCA
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def dim_reduce(train_features):
    # a simple PCA
    n_comp = 15
    pca = PCA(n_comp)
    model = pca.fit(train_features)

    '''
    # select an appropriate n_components
    plt.plot([x for x in range(100)], np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance Ratio")
    plt.savefig('cumulative_variance_ratio.png')
    # print(pca.explained_variance_ratio_)
    '''

    # extract most important features
    feature_size = train_features.shape[1]
    X_pc = model.transform(train_features)
    n_pcs = model.components_.shape[0]
    most_important = [np.abs(model.components_[i]).argmax() for i in range(n_pcs)]
    initial_feature_names = [x for x in range(feature_size + 1)]
    most_important_features = [initial_feature_names[most_important[i]] for i in range(n_pcs)]
    dic = {'PC{}'.format(i): most_important_features[i] for i in range(n_pcs)}

    most_important_features = set(most_important)
    selected_features = []
    for index in most_important_features:
        selected_features.append(list(train_features[:, index]))
    selected_features = np.array(selected_features)
    print("-------------------------------")
    print(selected_features)
    print("Total number features selected: {0}".format(selected_features.shape[0]))
    print("Total time slots for training: {0}".format(selected_features.shape[1]))

    #np.savetxt("security.csv", selected_features.T, delimiter=",")

    #for key in dic:
    #    print(dic[key])

    return selected_features, selected_features.shape[0]
