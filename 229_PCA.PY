import numpy as np
from format import get_data

class PCA(object):
	"""docstring for PCA"""
	'''
	data_matri
	data_matrix:m x n matrix. m: examples, n : features
	'''
	def __init__(self, data_matrix):
		super(PCA, self).__init__()
		print ("init PCA")
		self.data = data_matrix.astype(float)
		self.num_examples = len(data_matrix)
		self.num_features = len(data_matrix[0])
		self.num_eigen_vectors = self.num_examples

	def calculate_mu(self):
		mu_vector = np.zeros((self.num_features,))
		for ex_ind in range(self.num_examples):
			mu_vector = np.add(mu_vector, self.data[ex_ind])
		mu_vector = mu_vector/self.num_examples
		self.mu = mu_vector

	def subtract_mean_from_examples(self):
		f = lambda ex: np.subtract(ex, self.mu)
		for ex_ind in range(self.num_examples):
			self.data[ex_ind] = np.subtract(self.data[ex_ind], self.mu)

	def calc_variance(self):
		squared_sum = 0
		for ex_ind in range(self.num_examples):
			for feat_ind in range(self.num_features):
				squared_sum += self.data[ex_ind][feat_ind] ** 2
		self.variance = float(squared_sum) / self.num_examples

	def rescale_values_with_variance(self):
		for ex_ind in range(self.num_examples):
			for feat_ind in range(self.num_features):
				curr_val = self.data[ex_ind][feat_ind]
				rescaled_val = curr_val / float(self.variance)
				self.data[ex_ind][feat_ind] = rescaled_val

	def form_eigen_subparts(self):
                print(1)
		self.cov_matrix = np.cov(self.data.T)
                print(2)
		eigen_values, eigen_vectors = np.linalg.eig(self.cov_matrix)

		#sort eigen parts
                print(3)
		idx_arr = eigen_values.argsort()[::-1] #returns top num_example vectors
		print(4)
                eigen_values = eigen_values[idx_arr]
		print(5)
                eigen_vectors = eigen_vectors[:,idx_arr]
		print(6)
                self.top_eigen_vectors = []
		print(7)
                for ind in range(self.num_eigen_vectors):
			self.top_eigen_vectors.append(eigen_vectors[ind])

	def form_new_vectors(self):
		reduced_vectors = []
		# for ind in range(self.num_eigen_vectors):
		# 	curr_red_vec = np.dot(self.top_eigen_vectors[ind].T, 

		#form each new vector
		for ex_ind in range(self.num_examples):
			curr_red_vec = []
			for eig_ind in range(self.num_eigen_vectors):
				curr_red_vec.append(np.dot(self.top_eigen_vectors[eig_ind], 
											self.data[ex_ind]))
			reduced_vectors.append(curr_red_vec)
		print(reduced_vectors)
		return reduced_vectors


# my_arr = [[1, 2,5], [3,4,15]]
patients, egm_matrix, cancer_onehot = get_data()
print (egm_matrix.shape)
pca_class = PCA(egm_matrix)
pca_class.calculate_mu()
print ("calc mu")
pca_class.subtract_mean_from_examples()
print ("subtracted mean")
pca_class.calc_variance()
print("calculated variance")
pca_class.rescale_values_with_variance()
print ("rescale values")
pca_class.form_eigen_subparts()
print("formed eigen")
reduced_vectors = pca_class.form_new_vectors()
np.savetxt("pca_vectors.txt", reduced_vectors)
		
