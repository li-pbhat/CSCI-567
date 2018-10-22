import numpy as np
from typing import List, Set

from classifier import Classifier
from decision_stump import DecisionStump
from abc import abstractmethod

class Boosting(Classifier):
  # Boosting from pre-defined classifiers
	def __init__(self, clfs: Set[Classifier], T=0):
		self.clfs = clfs      # set of weak classifiers to be considered
		self.num_clf = len(clfs)
		if T < 1:
			self.T = self.num_clf
		else:
			self.T = T
	
		self.clfs_picked = [] # list of classifiers h_t for t=0,...,T-1
		self.betas = []       # list of weights beta_t for t=0,...,T-1
		return

	@abstractmethod
	def train(self, features: List[List[float]], labels: List[int]):
		return

	def predict(self, features: List[List[float]]) -> List[int]:
		'''
		Inputs:
		- features: the features of all test examples
   
		Returns:
		- the prediction (-1 or +1) for each example (in a list)
		'''
		########################################################
		# TODO: implement "predict"
		########################################################
		h = np.zeros(len(features))
		h = np.sum([beta * np.array(clf.predict(features)) for clf, beta in zip(self.clfs_picked, self.betas)], axis=0)
		h = np.sign(h)
		return h.tolist()
		

class AdaBoost(Boosting):
	def __init__(self, clfs: Set[Classifier], T=0):
		Boosting.__init__(self, clfs, T)
		self.clf_name = "AdaBoost"
		return
		
	def train(self, features: List[List[float]], labels: List[int]):
		'''
		Inputs:
		- features: the features of all examples
		- labels: the label of all examples
   
		Require:
		- store what you learn in self.clfs_picked and self.betas
		'''
		# implement "train"
		D = len(labels)
		w = np.full(D, 1 / D) # start with equal weights

		for t in range(self.T):
			et = np.inf
			# pick the classifier with least error for this iteration
			for clf in self.clfs:
				h = clf.predict(features)
				error = np.sum(w * (np.array(labels) != np.array(h)))
				if error < et:
					ht = clf
					et = error
					htx = h
			self.clfs_picked.append(ht)

			beta = 0.5 * np.log((1 - et) / et)
			self.betas.append(beta)

			# update weights based on the new value of beta
			for n in range(D):
				w[n] *= np.exp(-beta) if labels[n] == htx[n] else np.exp(beta)

			# normalize weights
			w /= np.sum(w)

		
		
	def predict(self, features: List[List[float]]) -> List[int]:
		return Boosting.predict(self, features)



	