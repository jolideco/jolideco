import numpy as np
from .gmm import GaussianMixtureModel

__all__ = ["GMMNode", "split"]


class GMMNode:
    """Binary tree GMM 
    
    Attributes
    ----------
    left : `GMMNode`
        Left GMM node
    right : `GMMNode`
        Right GMM node
    gmm : `~GaussianMixtureModel`
        Gaussian mixture model
    
    """
    def __init__(self, left=None, right=None, gmm=None):
        if gmm and not gmm.n_components == 1:
            raise ValueError("GaussianMixtureModel can only have one component")
            
        self.left = left
        self.right = right
        self._gmm = gmm
    
    @property
    def is_base_node(self):
        """Is base node"""
        return (self.left is None) and (self.right is None)

    @property
    def gmm(self):
        """Merged GMM components, by computing a weighted average.
        
        Returns
        -------
        gmm : `~GaussianMixtureModel`
            Gaussian mixture model
        """
        if self._gmm is None:        
            weights = self.left.gmm.weights + self.right.gmm.weights

            covariances = self.left.gmm.weights * self.left.gmm.covariances
            covariances += self.right.gmm.weights * self.right.gmm.covariances

            means = self.left.gmm.weights * self.left.gmm.means
            means += self.right.gmm.weights * self.right.gmm.means

            self._gmm = GaussianMixtureModel(
                covariances=covariances / weights,
                means=means / weights,
                weights=weights
            )
        return self._gmm
    
    def estimate_log_prob_max(self, x):
        """Compute max. log likelihood for a given feature vector"""
        if self.is_base_node:
            value = self.gmm.estimate_log_prob(x=x)
        else:
            value_left = self.left.gmm.estimate_log_prob(x=x)
            value_right = self.right.gmm.estimate_log_prob(x=x)

            evaluate_left = (value_left > value_right)
            evaluate_right = ~evaluate_left

            value = np.empty(value_left.shape)
            
            if evaluate_left.any():
                p =  self.left.estimate_log_prob_max(x=x[evaluate_left[:, 0], :])
                value[evaluate_left[:, 0], :] = p
            
            if evaluate_right.any():
                p = self.right.estimate_log_prob_max(x=x[evaluate_right[:, 0], :])
                value[evaluate_right[:, 0], :] = p
        
        return value


class BTGaussianMixtureModel(GMMNode):
    """Binary tree Gaussian mixture model"""
    
    @classmethod
    def from_gaussian_mixture_model(cls, gmm):
        """Create bmary tree GMM from normal GMM

        Parameters
        ----------
        gmm : `~GaussianMixtureModel`
            Gaussian mixture model
            
        Returns
        -------
        bt_gmm : `BTGaussianMixtureModel`
            Binary tree Gaussian mixture model
        """
        nodes = split_gmm_into_nodes(gmm=gmm)
        root = build_gmm_binary_tree(nodes=nodes)
        return cls(left=root.left, right=root.right)
    

def split_gmm_into_nodes(gmm):
    """Split a GMM into many nodes
    
    Parameters
    ----------
    gmm : `~GaussianMixtureModel`
        Gaussian mixture model
        
    Returns
    -------
    nodes : list of `GMMNodes`
        GMM nodes
    """
    nodes = []
    
    for idx in range(gmm.n_components):
        idx = slice(idx, idx + 1)
        gmm_node = GaussianMixtureModel(
            means=gmm.means[idx],
            weights=gmm.weights[idx],
            covariances=gmm.covariances[idx],
        )
        node = GMMNode(gmm=gmm_node)
        nodes.append(node)
    
    return nodes


def find_best_pairs(nodes, n_iter=10_000):
    """Find best pairs by means of the KL divergence"""
    indices = np.arange(len(nodes))
    kl_best, pairs_best = np.inf, None

    for idx in range(n_iter):
        indices = np.random.permutation(indices)
        pairs = indices.reshape((-1, 2))
        kl_total = 0

        for idx, idy in pairs:
            gmm, gmm_other = nodes[idx].gmm, nodes[idy].gmm
            kl_value = gmm.symmetric_kl_divergence(gmm_other)
            kl_total += kl_value
        
        if kl_total < kl_best:
            kl_best = kl_total
            pairs_best = pairs

    return pairs_best



def build_gmm_binary_tree(nodes):
    """Build binary tree
    
    Parameters
    ----------
    nodes : list of `GMMNode`
        GMM nodes

    Returns
    -------
    root : `GMMNode`
        Root GMMNode
    """
    n_nodes = len(nodes)
    
    for level in range(int(np.log2(n_nodes))):
        pairs = find_best_pairs(nodes=nodes)
    
        nodes_merged = []
        
        for idx, idy in pairs:
            node = GMMNode(
                left=nodes[idx], right=nodes[idy]
            )
            nodes_merged.append(node)
    
        nodes = nodes_merged
        
    return nodes[0]
    