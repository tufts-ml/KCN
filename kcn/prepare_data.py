import numpy as np
import scipy
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.neighbors import NearestNeighbors
from sklearn.model_selection import train_test_split


def coords_to_kgraph(coords, k, kernel_length):
    """
    Find k nearest neighbors of data points by GPS locations

    :param k: the number of neighbors
    :param coords: The gps coordinates N X 2 .
    :return: The M X K nearest neighbors matrix.
    """

    nbrs = NearestNeighbors(n_neighbors=k).fit(coords)
    graph = nbrs.kneighbors_graph(mode='distance')
    graph.data = np.exp(-0.5 * graph.data * graph.data / (kernel_length * kernel_length))

    return graph


def load_data(name, n_neighbors=5):
    if name == 'birds' or name == 'yelp_stars' or name == 'linfeng_precip' or name == 'housing_prices':
        # TODO: revise the data path if needed
        if name == 'birds':
            # read in data from the file
            file_path = '../gcn/birds.npz'
        elif name == 'linfeng_precip':
            file_path = '../gcn/linfeng_precip.npz'
        elif name == 'housing_prices':
            file_path = '../../data/housing_prices.npz'
        elif name == 'yelp_stars':
            file_path = '../gcn/yelp_stars.npz'
        else:
            raise Exception('No such dataset' + name)

        data = np.load(file_path)
        coords_train = np.ndarray.astype(data['Xtrain'], np.float32)
        coords_test = np.ndarray.astype(data['Xtest'], np.float32)

        values_train = data['Ytrain']
        values_test = data['Ytest']

        # check and record shapes
        assert (coords_train.shape[0] == values_train.shape[0])
        assert (coords_test.shape[0] == values_test.shape[0])

        Ntrain = coords_train.shape[0]
        Ntest = coords_test.shape[0]
        Nval = 1000
        N = Ntrain + Ntest

        rand_ind = np.random.permutation(Ntrain)
        rand_ind = np.random.permutation(Ntrain)
        coords_train = coords_train[rand_ind]
        values_train = values_train[rand_ind]

        # merge train and test # get features and coordinates
        coords_features = np.concatenate([coords_train, coords_test], axis=0)
        coords = coords_features[:, 0:2]
        features = coords_features

        # feature normalization
        fmean = np.mean(features[0:Ntrain], axis=0, keepdims=True)
        fstd = np.std(features[0:Ntrain], axis=0, keepdims=True)
        features = (features - fmean) / (fstd + 0.01)

        # features =  scipy.sparse.lil_matrix(features)

        y = np.concatenate([values_train, values_test], axis=0)
        y = y[:, None]

        if name == 'linfeng_precip':
            y = np.log(y)
        #
        # if name == 'yelp_stars':
        #     # concat 0 as variance
        #     zeros = np.zeros_like(y)
        #     y = np.concatenate([y, zeros], axis=1)

        # get masks

        train_mask = np.zeros(N, dtype=bool)
        train_mask[0:(Ntrain - 500)] = True

        val_mask = np.zeros(N, dtype=bool)
        val_mask[(Ntrain - 500):Ntrain] = True

        test_mask = np.zeros(N, dtype=bool)
        test_mask[Ntrain:N] = True

        y_train = y.copy()
        y_train[np.logical_not(train_mask), 0] = 0

        y_val = y.copy()
        y_val[np.logical_not(val_mask), 0] = 0

        y_test = y.copy()
        y_test[np.logical_not(test_mask), 0] = 0

        ## getting neighbors from training data and validation data.
        # knn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords[0:Ntrain])
        # _, nbs_train_val = knn.kneighbors()
        # _, nbs_test = knn.kneighbors(coords[Ntrain:])

        # nbs = np.concatenate([nbs_train_val, nbs_test], axis=0)

        ## concatenate index of itself
        # nbs = np.concatenate([np.arange(nbs.shape[0])[:, None], nbs], axis=1)


    else:
        raise Exception('No such datasets: %s' % name)

    return coords, features, y_train, y_val, y_test, train_mask, val_mask, test_mask  # , nbs


def load_kriging_data(file_path, n_neighbors):
    data = np.load(file_path)
    coords_train = np.ndarray.astype(data['Xtrain'], np.float32)
    coords_test = np.ndarray.astype(data['Xtest'], np.float32)
    values_train = data['Ytrain']
    values_test = data['Ytest']

    # check and record shapes
    assert (coords_train.shape[0] == values_train.shape[0])
    assert (coords_test.shape[0] == values_test.shape[0])

    Ntrain = coords_train.shape[0]
    Ntest = coords_test.shape[0]
    N = Ntrain + Ntest

    rand_ind = np.random.permutation(Ntrain)
    coords_train = coords_train[rand_ind]
    values_train = values_train[rand_ind]

    # merge train and test # get features and coordinates
    coords_features = np.concatenate([coords_train, coords_test], axis=0)
    coords = coords_features[:, 0:2]
    features = coords_features

    # feature normalization
    fmean = np.mean(features[0:Ntrain], axis=0, keepdims=True)
    fstd = np.std(features[0:Ntrain], axis=0, keepdims=True)
    features = (features - fmean) / (fstd + 0.01)

    # features =  scipy.sparse.lil_matrix(features)

    y = np.concatenate([values_train, values_test], axis=0)
    y = y[:, None]

    # get masks
    y_train_val = y.copy()
    y_train_val[Ntrain:] = 0

    # get masks
    train_mask = np.zeros(N, dtype=bool)
    train_mask[0:(Ntrain - 500)] = True

    val_mask = np.zeros(N, dtype=bool)
    val_mask[(Ntrain - 500):Ntrain] = True

    test_mask = np.zeros(N, dtype=bool)
    test_mask[Ntrain:N] = True

    # getting neighbors from training data and validation data.
    knn = NearestNeighbors(n_neighbors=n_neighbors).fit(coords[0:Ntrain])
    _, nbs_train_val = knn.kneighbors()
    _, nbs_test = knn.kneighbors(coords[Ntrain:])

    nbs = np.concatenate([nbs_train_val, nbs_test], axis=0)

    # concatenate index of itself
    nbs = np.concatenate([np.arange(nbs.shape[0])[:, None], nbs], axis=1)

    return coords, features, y, y_train_val, nbs, Ntrain, train_mask, val_mask, test_mask


if __name__ == '__main__':
    load_data('bird_counts')

    print('Done!')
