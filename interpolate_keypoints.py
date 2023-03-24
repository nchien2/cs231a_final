import torch
import numpy as np

def interpolate_vector(x):
	a = x.astype(float)
	a_len = a.size
	if (a == np.zeros(a_len)).all(): return a
	zero_pos = np.where(a == 0.0)[0]
	z_len = zero_pos.size
	segments = [] # list of tuples of the start and end of indices of consecutive zeros
	i = 0
	while i < z_len:
		j = i
		while j + 1 < z_len and j + 1 - i == zero_pos[j + 1] - zero_pos[i]: j += 1
		segments.append((zero_pos[i], zero_pos[j]))
		i = j + 1
	
	for seg in segments:
		start = seg[0]
		end = seg[1]
		if start == 0:
			a[start:end + 1] = a[end + 1] * np.ones(end - start + 1)
		elif end == a_len - 1:
			a[start:end + 1] = a[start - 1] * np.ones(end - start + 1)
		else:
			a[start:end + 1] = np.linspace(a[start - 1], a[end + 1], end - start + 3)[1:-1]

	return a



# takes in m (num frames) by n (num feat) by 2 (x and y)
# tensor and interpolates where zeros are found
def interpolate_keypoints(data):
	m, n, d = data.shape
	assert(d == 2)
	result = torch.clone(data)
	for j in range(n):
		for k in range(d):
			result[:, j, k] = torch.tensor(interpolate_vector(np.array(data[:, j, k])))
	return result


if __name__ == "__main__":
	a = np.array([1,0,0,0,0,0,0,0,3])
	print(interpolate_vector(a))
	a = np.array(range(15)).reshape((5, 3)).astype(float)
	a[1:3, :2] = np.zeros((2, 2))
	a[3, 2] = 0
	print(a)
	M = torch.tensor(np.array([a, a])).reshape(5, 3, 2)
	print(M)
	print(interpolate_keypoints(M))