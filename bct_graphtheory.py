# Taku Ito
# 08/26/2014
"""
Python implementation of some graph-theoretic measures modeled after the Brain Connectivity Toolbox.
Original MatLab implementation found:
https://github.com/fieldtrip/fieldtrip/blob/master/external/bct/


Complex network measures of brain connectivity: Uses and interpretations.
Rubinov M, Sporns O (2010) NeuroImage 52:1059-69.
Web address: http://www.brain-connectivity-toolbox.net
"""

import numpy as np

def threshold_proportional(W, p):
	"""
	Original BCT MatLab description:

	THRESHOLD_PROPORTIONAL     Proportional thresholding

	W_thr = threshold_proportional(W, p);

	This function "thresholds" the connectivity matrix by preserving a
	proportion p (0<p<1) of the strongest weights. All other weights, and
	all weights on the main diagonal (self-self connections) are set to 0.

	Inputs: W,      weighted or binary connectivity matrix (as an numpy, ndarray)
	       p,      proportion of weights to preserve
	                   range:  p=1 (all weights preserved) to
	                           p=0 (no weights removed)

	Output: W_thr,  thresholded connectivity matrix


	Mika Rubinov, UNSW, 2010

	Python Implementation: Taku Ito, 2014
	"""

	# Get number of nodes
	n = len(W)

	# Clear diagonal
	np.fill_diagonal(W, 0)

	###
	# Sort connection strengths (doesn't absolute value connections)! If only interested for absolute values, absolute value matrix prior to input.
	# Reshape connection strengths to 1d array
	reshapeMat = W.reshape(-1).copy()
	# Enumerate connections
	indices = np.arange(len(W)**2)
	# Stack indices and connection values horizontally
	sortTuple = np.column_stack((indices, reshapeMat))
	# Sort in descending order
	sortTuple = sortTuple[np.argsort(-sortTuple[:,1])]
	# Remove all 0 connections (since 0 connections are trivial diagonal values)
	sortTuple = sortTuple[sortTuple[:,1]!=0]
	###

	# number of links to be preserved
	en = round((n**2-n)*p)
	# Get indices of links to be thrown out
	throwout = sortTuple[en:,0].astype(int)
	# Set throwout connections to 0
	reshapeMat[throwout] = 0
	# Reshape back to symmetric matrix
	tW = reshapeMat.reshape(n,n)

	# Verify that matrix is still symmetric
	if (tW.transpose() != tW).all() == True:
		raise Exception('Something is wrong! Matrix not symmetric.')

	return tW


def strengths_und_sign(W):
	"""
	Original BCT MatLab description:

	STRENGTHS_UND_SIGN        Strength and weight
 
    [Spos Sneg] = strengths_und_sign(W);
    [Spos Sneg vpos vneg] = strengths_und_sign(W);
 
    Node strength is the sum of weights of links connected to the node.
 
    Inputs:     W,              undirected connection matrix with positive
                                and negative weights
 
    Output:     Spos/Sneg,      nodal strength of positive/negative weights
                vpos/vneg,      total positive/negative weight
 
 
    2011, Mika Rubinov, UNSW

	Python Implementation: Taku Ito, 2014

	"""

	# Number of nodes
	n = len(W)

	# Clear Diagonal
	np.fill_diagonal(W, 0)
	Spos = np.sum(np.multiply(W,W>0),axis=0)
	Sneg = np.sum(np.multiply(W,W<0),axis=0)

	vpos = np.sum(Spos)
	vneg = np.sum(Sneg)

	return Spos, Sneg, vpos, vneg