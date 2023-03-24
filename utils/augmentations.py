# AUGMENTATIONS
import random
import math
import torch
import pandas as pd
import numpy as np
import os

# Any argument named keypoints should be a 1D tensor of keypoint values, eg keypoint_json['pose_keypoints_2d']

def keypoint_to_coord(keypoints):
  keypoints = keypoints.reshape(-1, 3)
  uncertainty = keypoints[:,2]
  x = keypoints[:,0]
  y = keypoints[:,1]
  return x, y, uncertainty

def coord_to_keypoint(x, y, uncertainty):
  coord_tensor = torch.stack((x, y, uncertainty), dim=1)
  keypoints = coord_tensor.reshape(-1)
  return keypoints

# Rotate around center by a random angle
def rotation(keypoints):
  angle = random.uniform(-0.15, 0.15)
  x, y, uncertainties = keypoint_to_coord(keypoints)
  
  x_rot = (x ) * math.cos(angle) - (y) * math.sin(angle)
  y_rot = (y) * math.cos(angle) + (x) * math.sin(angle)
  keypoints = coord_to_keypoint(x_rot, y_rot, uncertainties)
  return keypoints

# 
def squeeze(keypoints):
  x, y, uncertainties = keypoint_to_coord(keypoints)

  width = torch.max(x) - torch.min(x)
  left_squeeze = random.uniform(0, 0.1) * width
  right_squeeze = random.uniform(0, 0.1) * width
  x = (x - left_squeeze) / (width - left_squeeze - right_squeeze)
  keypoints = coord_to_keypoint(x, y, uncertainties)
  return keypoints

def projection(keypoints):
  x, y, uncertainties = keypoint_to_coord(keypoints)
  rot_angle = np.random.uniform(-0.15, 0.15)
  Tx = np.random.uniform(-0.15, 0.15)
  Ty = np.random.uniform(-0.15, 0.15)
  R = torch.tensor([
        [math.cos(rot_angle), -math.sin(rot_angle), Tx],
        [math.sin(rot_angle), math.cos(rot_angle),Ty],
        [0, 0, 1]
  ])

  Sx = np.random.uniform(-0.1, 0.1)
  Sy = np.random.uniform(-0.1, 0.1)
  A = torch.tensor([
    [1, Sy, 0],
    [Sx, 1, 0],
    [0,0,1]
  ])

  p1 = np.random.uniform(-0.0001, 0.0001)
  p2 = np.random.uniform(-0.0001, 0.0001)
  P = torch.tensor([
      [1, 0, 0],
      [0, 1, 0],
      [p1,p2,1]
  ])
  H = R @ A @ P 
  coords = torch.stack((x, y, torch.ones(x.shape[0])))
  new_coords = H @ coords
  new_coords = new_coords[:2, :] / new_coords[2,:]
  x = new_coords[0, :]
  y = new_coords[1, :]
  keypoints_new = coord_to_keypoint(x, y, uncertainties)
  return keypoints_new


def augment(keypoints):
  p = np.random.randint(0, 3)
  if p == 0:
    keypoints = rotation(keypoints)
  elif p == 1:
    keypoints = squeeze(keypoints)
  elif p == 2:
    keypoints = projection(keypoints)
  return keypoints


# NORMALIZATION CODE =============================================
def normalize_helper(keypoints):
  x, y, uncertainties = keypoint_to_coord(keypoints)
  x = x.float()
  y = y.float()

  x = (x - torch.mean(x)) / torch.std(x)
  y = (y - torch.mean(y)) / torch.std(y)

  keypoints = coord_to_keypoint(x, y, uncertainties)
  return keypoints


def normalize_keypoints(kp_json):
  left_hand = torch.tensor(kp_json['hand_left_keypoints_2d'])
  right_hand = torch.tensor(kp_json['hand_right_keypoints_2d'])
  face = torch.tensor(kp_json['face_keypoints_2d'])
  body = torch.tensor(kp_json['pose_keypoints_2d'])

  normalized = {}
  normalized['hand_left_keypoints_2d'] = normalize_helper(left_hand).tolist()
  normalized['hand_right_keypoints_2d'] = normalize_helper(right_hand).tolist()
  normalized['face_keypoints_2d'] = normalize_helper(face).tolist()
  normalized['pose_keypoints_2d'] = normalize_helper(body).tolist()

  return normalized


# LOOP THROUGH ALL FILES AND NORMALIZE FOR PRE-PROCESSING

def normalize_all():
  annotations_file = 'drive/MyDrive/CS231A/asllvd_signs_2023_02_16.csv'
  pose_dir = 'drive/MyDrive/CS231A/ASLLVD/output_json/'
  output_dir = 'drive/MyDrive/CS231A/ASLLVD/normalized_json'

  df = pd.read_csv(annotations_file)
  df = df.iloc[9500:]


  for idx, row in df.iterrows():
    pose_path = os.path.join(pose_dir, f'{idx}')
    output_path = os.path.join(output_dir, f'{idx}')

    if not os.path.isdir(pose_path):
      continue
    elif not os.path.isdir(output_path):
      os.makedirs(output_path)

    for filename in os.listdir(pose_path):
      json_path = os.path.join(pose_path, filename)
      f = open(json_path)
      kp_json = json.load(f)['people'][0]
      kp_json = normalize_keypoints(kp_json)

      with open(os.path.join(output_path, filename),'w') as output_f:
        json.dump(kp_json, output_f, ensure_ascii=False, indent=4)


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