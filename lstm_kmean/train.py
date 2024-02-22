import tensorflow as tf
import numpy as np
from glob import glob
from natsort import natsorted
import os
import pickle
from model import TripleNet, train_step, test_step
from utils import load_complete_data
from eeg2_dataloaders import create_EEG_dataset, create_image_dataset
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from matplotlib import style
# import seaborn as sns
import pandas as pd
from sklearn.cluster import KMeans

# style.use('seaborn')

os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

np.random.seed(45)
tf.random.set_seed(45)

if __name__ == '__main__':

	# Using EEG2 dataset
	# n_channels  = 17
	# n_feat      = 128
	# batch_size  = 256
	# test_batch_size  = 1
	# n_classes   = 100
	# dataset_path = "C:\\Users\\Nick\\Repos\\EEG2Image\\eeg2_data"
	# subjects = ["sub-01"]
	# print('Loading the eeg dataset:...')
	# eeg_train_data, eeg_test_data = create_EEG_dataset(dataset_path, subjects)
	# train_X = np.expand_dims(np.mean(eeg_train_data['sub-01']['preprocessed_eeg_data'], axis=1), axis=3)[0:1000]
	# train_Y = np.zeros((1000, 100), dtype=int)
	# for row in range(train_Y.shape[0]):
	# 	train_Y[row, int(row / 10)] = 1
	# train_batch = load_complete_data(train_X, train_Y, batch_size=batch_size)
	# val_batch   = load_complete_data(train_X, train_Y, batch_size=batch_size)
	# test_batch  = load_complete_data(train_X, train_Y, batch_size=test_batch_size)
	

	n_channels  = 14
	n_feat      = 128
	batch_size  = 256
	test_batch_size  = 1
	n_classes   = 10

	# data_cls = natsorted(glob('data/thoughtviz_eeg_data/*'))
	# cls2idx  = {key.split(os.path.sep)[-1]:idx for idx, key in enumerate(data_cls, start=0)}
	# idx2cls  = {value:key for key, value in cls2idx.items()}

	with open('data/eeg/char/data.pkl', 'rb') as file:
		data = pickle.load(file, encoding='latin1')
		train_X = data['x_train']
		train_Y = data['y_train']
		test_X = data['x_test']
		test_Y = data['y_test']

	# # train_batch = load_complete_data('data/thoughtviz_eeg_data/*/train/*', batch_size=batch_size)
	# # val_batch   = load_complete_data('data/thoughtviz_eeg_data/*/val/*', batch_size=batch_size)
	# # test_batch  = load_complete_data('data/thoughtviz_eeg_data/*/test/*', batch_size=test_batch_size)
	train_batch = load_complete_data(train_X, train_Y, batch_size=batch_size)
	val_batch   = load_complete_data(test_X, test_Y, batch_size=batch_size)
	test_batch  = load_complete_data(test_X, test_Y, batch_size=test_batch_size)
	
	X, Y = next(iter(train_batch))

	# print(X.shape, Y.shape)
	triplenet = TripleNet(n_classes=n_classes)
	opt     = tf.keras.optimizers.Adam(learning_rate=3e-4)
	triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
	triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='experiments/best_ckpt', max_to_keep=5000)
	triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint)
	# START = int(triplenet_ckpt.step) // len(train_batch)
	START = 3000
	# if triplenet_ckptman.latest_checkpoint:
	# 	print('Restored from the latest checkpoint, epoch: {}'.format(START))
	EPOCHS = 0
	# cfreq  = 1 # Checkpoint frequency
	smallest_loss = 0.95
	for epoch in range(START, EPOCHS):
		train_acc  = tf.keras.metrics.SparseCategoricalAccuracy()
		train_loss = tf.keras.metrics.Mean()
		test_acc   = tf.keras.metrics.SparseCategoricalAccuracy()
		test_loss  = tf.keras.metrics.Mean()

		tq = tqdm(train_batch)
		for idx, (X, Y) in enumerate(tq, start=1):
			loss = train_step(triplenet, opt, X, Y)
			train_loss.update_state(loss)
			# Y_cap = triplenet(X, training=False)
			# train_acc.update_state(Y, Y_cap)
			triplenet_ckpt.step.assign_add(1)
			# tq.set_description('Train Epoch: {}, Loss: {}, Acc: {}'.format(epoch, train_loss.result(), train_acc.result()))
			tq.set_description('Train Epoch: {}, Loss: {}'.format(epoch, train_loss.result()))
			# break

		tq = tqdm(val_batch)
		test_loss_result = float("inf")
		for idx, (X, Y) in enumerate(tq, start=1):
			loss = test_step(triplenet, X, Y)
			test_loss.update_state(loss)
			# Y_cap = triplenet(X, training=False)
			# test_acc.update_state(Y, Y_cap)
			# tq.set_description('Test Epoch: {}, Loss: {}'.format(epoch, test_loss.result(), test_acc.result()))
			tq.set_description('Val Epoch: {}, Loss: {}'.format(epoch, test_loss.result()))
			test_loss_result = float(test_loss.result())
			# break
		if test_loss_result < smallest_loss:
				triplenet_ckptman.save()
				smallest_loss = test_loss_result

	kmeanacc = 0.0
	tq = tqdm(test_batch)
	feat_X = []
	feat_Y = []
	for idx, (X, Y) in enumerate(tq, start=1):
		_, feat = triplenet(X, training=False)
		feat_X.extend(feat.numpy())
		feat_Y.extend(Y.numpy())
	feat_X = np.array(feat_X)
	feat_Y = np.array(feat_Y)
	print(feat_X.shape, feat_Y.shape)
	# colors = list(plt.cm.get_cmap('viridis', 10))
	# print(colors)
	# colors  = [np.random.rand(3,) for _ in range(10)]
	# print(colors)
	# Y_color = [colors[label] for label in feat_Y]

	tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=700)
	tsne_results = tsne.fit_transform(feat_X)
	df = pd.DataFrame()
	df['label'] = feat_Y
	df['x1'] = tsne_results[:, 0]
	df['x2'] = tsne_results[:, 1]
	# df['x3'] = tsne_results[:, 2]
	df.to_csv('experiments/inference/triplet_embed2D.csv')
	
	# df.to_csv('experiments/triplenet_embed3D.csv')
	# df = pd.read_csv('experiments/triplenet_embed2D.csv')
	
	df = pd.read_csv('experiments/inference/triplet_embed2D.csv')

	plt.figure(figsize=(16,10))
	
	# ax = plt.axes(projection='3d')

	custom_labels = ['a', 'c', 'f', 'h', 'j', 'm', 'p', 's', 't', 'y']

	sns.scatterplot(
		x="x1", y="x2",
		data=df,
		hue='label',
		hue_order=range(n_classes),  # Ensure the order matches the number of classes
		palette=sns.color_palette("hls", n_classes),
		legend="full",
		alpha=0.4
	)

	# Set custom labels for the legend
	plt.legend(labels=custom_labels)

	plt.show()
	# plt.savefig('experiments/inference/{}_embedding.png'.format(epoch))

	kmeans = KMeans(n_clusters=n_classes,random_state=45)
	kmeans.fit(feat_X)
	labels = kmeans.labels_
	# print(feat_Y, labels)
	correct_labels = sum(feat_Y == labels)
	print("Result: %d out of %d samples were correctly labeled." % (correct_labels, feat_Y.shape[0]))
	kmeanacc = correct_labels/float(feat_Y.shape[0])
	print('Accuracy score: {0:0.2f}'. format(kmeanacc))

	# with open('experiments/triplenet_log.txt', 'a') as file:
	# 	file.write('E: {}, Train Loss: {}, Test Loss: {}, KM Acc: {}\n'.\
	# 		format(epoch, train_loss.result(), test_loss.result(), kmeanacc))
	# break