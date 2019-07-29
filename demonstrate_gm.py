import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split
from sklearn import mixture
from gaussian_mixture import GaussianMixture


# get a random generated data
def get_data(visualize=True):
	with open('data.pickle', 'rb') as f:
		data_points = pickle.load(f)
	if visualize:
		plt.scatter(data_points[:, 0], data_points[:, 1])
		plt.xlabel("Feature 0")
		plt.ylabel("Feature 1")
		plt.show()
	return data_points


x = get_data()
x_train, x_test = train_test_split(x, test_size=0.1, random_state=1)

gm = GaussianMixture(k=2)
gm.fit(x_train)
predictions = gm.predict(x_test)

sk_gm = mixture.GaussianMixture(n_components=2)
sk_gm.fit(x_train)
sk_predictions = sk_gm.predict(x_test)

print(f'Custom implementation predictions: {predictions}')
print(f'SKlearn implementation predictions: {sk_predictions}')
