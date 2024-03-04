import tensorflow as tf
# from utils import vis, load_batch#, load_data
from utils import load_complete_data, show_batch_images
from model import DCGAN, dist_train_step#, train_step
from tqdm import tqdm
import os
import shutil
import pickle
from glob import glob
from natsort import natsorted
import wandb
import numpy as np
import cv2
from lstm_kmean.model import TripleNet
import math
# from eval_utils import get_inception_score
tf.random.set_seed(45)
np.random.seed(45)

# clstoidx = {}
# idxtocls = {}

# for idx, item in enumerate(natsorted(glob('data/charimages/train/*')), start=0):
# 	clsname = os.path.basename(item)
# 	clstoidx[clsname] = idx
# 	idxtocls[idx] = clsname

# image_paths = natsorted(glob('data/charimages/train/*/*'))
# imgdict     = {}
# for path in image_paths:
# 	key = path.split(os.path.sep)[-2]
# 	if key in imgdict:
# 		imgdict[key].append(path)
# 	else:
# 		imgdict[key] = [path]

# wandb.init(project='DCGAN_DiffAug_EDDisc_imagenet_128', entity="prajwal_15")

def load_brain2image_dataset(dataset_path):
    eeg_data_list = []
    class_labels_list = []
    images_list = []
    class_name_to_label = {}  # Maps class names to integers

    # Iterate over the files in the directory
    for file_name in os.listdir(dataset_path):
        if file_name.endswith('.npy') and file_name[-5] == '1': # Only load first subject's data
            # Construct the full file path
            file_path = os.path.join(dataset_path, file_name)
            # Load the .npy file
            data = np.load(file_path, allow_pickle=True)

            # Extract EEG data (assuming it's the second element in the array)
            eeg_data = data[1]
            eeg_data_list.append(eeg_data)

            # Extract the image (assuming it's the first element in the array)
            image = data[0]
            image = np.float32(cv2.resize(image, (128, 128)))
            image = np.transpose(image, (1, 0, 2))
            images_list.append(image)

            # Extract the class label from the file name
            class_name = file_name.split('_')[0]
            if class_name not in class_name_to_label:
                # Assign a new label to this class name if it's the first time we see it
                class_name_to_label[class_name] = len(class_name_to_label)
            class_label = class_name_to_label[class_name]
            class_labels_list.append(class_label)

    # Convert lists to numpy arrays
    eeg_data_array = np.expand_dims(np.array(eeg_data_list), axis=3)
    class_labels_array = np.array(class_labels_list)
    images_array = np.array(images_list)

    number_of_classes = 40  # Total number of classes

    class_labels_array = np.array(class_labels_list)  # Convert list to NumPy array

    one_hot_labels = np.zeros((class_labels_array.size, number_of_classes))

    one_hot_labels[np.arange(class_labels_array.size), class_labels_array] = 1

    return eeg_data_array, one_hot_labels, images_array

os.environ['TF_XLA_FLAGS'] = '--tf_xla_enable_xla_devices'
os.environ["CUDA_DEVICE_ORDER"]= "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]= '0'

if __name__ == '__main__':
	n_channels = 128
	n_features = 128
	batch_size = 40
	test_batch_size = 1
	n_classes = 40
	# n_channels  = 14
	# n_feat      = 128
	# batch_size  = 128
	# test_batch_size  = 1
	# n_classes   = 10

	# data_cls = natsorted(glob('data/thoughtviz_eeg_data/*'))
	# cls2idx  = {key.split(os.path.sep)[-1]:idx for idx, key in enumerate(data_cls, start=0)}
	# idx2cls  = {value:key for key, value in cls2idx.items()}

	# with open('data/eeg/char/data.pkl', 'rb') as file:
	# 	data = pickle.load(file, encoding='latin1')
	# 	train_X = data['x_train']
	# 	train_Y = data['y_train']
	# 	test_X = data['x_test']
	# 	test_Y = data['y_test']

	# train_path = []
	# for X, Y in zip(train_X, train_Y):
	# 	train_path.append(np.random.choice(imgdict[idxtocls[np.argmax(Y)]], size=(1,) ,replace=True)[0])

	# test_path = []
	# for X, Y in zip(test_X, test_Y):
	# 	test_path.append(np.random.choice(imgdict[idxtocls[np.argmax(Y)]], size=(1,) ,replace=True)[0])

	directory_path_train = '/workspace/shared/eegstylegan/eeg_imagenet40_cvpr_2017_raw/train/'
	train_X, train_Y, train_I = load_brain2image_dataset(directory_path_train)
	directory_path_test = '/workspace/shared/eegstylegan/eeg_imagenet40_cvpr_2017_raw/test/'
	test_X, test_Y, test_I = load_brain2image_dataset(directory_path_test)

	train_batch = load_complete_data(train_X, train_Y, train_I, batch_size=batch_size)
	test_batch  = load_complete_data(test_X, test_Y, test_I, batch_size=test_batch_size)
	X, Y, I      = next(iter(train_batch))
	latent_label = Y[:16]

	gpus = tf.config.list_physical_devices('GPU')
	mirrored_strategy = tf.distribute.MirroredStrategy(devices=['/GPU:1'], 
		cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
	n_gpus = mirrored_strategy.num_replicas_in_sync
	# print(n_gpus)

	# batch_size = 64
	latent_dim = 128
	input_res  = 128

	# print(latent_Y)
	# latent_Y = latent_Y[:16]
	# print

	triplenet = TripleNet(n_classes=n_classes)
	opt = tf.keras.optimizers.legacy.Adam(learning_rate=3e-4)
	triplenet_ckpt    = tf.train.Checkpoint(step=tf.Variable(1), model=triplenet, optimizer=opt)
	triplenet_ckptman = tf.train.CheckpointManager(triplenet_ckpt, directory='lstm_kmean/experiments/best_ckpt', max_to_keep=5000)
	triplenet_ckpt.restore(triplenet_ckptman.latest_checkpoint)
	print('TripletNet restored from the latest checkpoint: {}'.format(triplenet_ckpt.step.numpy()))
	_, latent_Y = triplenet(X, training=False)

	print('Extracting test eeg features:')
	# test_eeg_features = np.array([np.squeeze(triplenet(E, training=False)[1].numpy()) for E, Y, X in tqdm(test_batch)])
	# test_eeg_y        = np.array([Y.numpy()[0] for E, Y, X in tqdm(test_batch)])
	test_image_count = 50000 #// n_classes
	# test_labels = np.tile(np.expand_dims(np.arange(0, 10), axis=-1), [1, test_image_count//n_classes])
	# test_labels = np.sort(test_labels.ravel())

	test_eeg_cls      = {}
	for E, Y, X in tqdm(test_batch):
		Y = Y.numpy()[0]
		if Y not in test_eeg_cls:
			test_eeg_cls[Y] = [np.squeeze(triplenet(E, training=False)[1].numpy())]
		else:
			test_eeg_cls[Y].append(np.squeeze(triplenet(E, training=False)[1].numpy()))

	for _ in range(n_classes):
		test_eeg_cls[_] = np.array(test_eeg_cls[_])
		print(test_eeg_cls[_].shape)

	for cl in range(n_classes):
		N = test_eeg_cls[cl].shape[0]
		per_cls_image = int(math.ceil((test_image_count//n_classes) / N))
		test_eeg_cls[cl] = np.expand_dims(test_eeg_cls[cl], axis=1)
		test_eeg_cls[cl] = np.tile(test_eeg_cls[cl], [1, per_cls_image, 1])
		test_eeg_cls[cl] = np.reshape(test_eeg_cls[cl], [-1, latent_dim])
		print(test_eeg_cls[cl].shape)

	# test_image_count = test_image_count // n_classes
	# print(test_eeg_features.shape, test_eeg_y.shape)

	lr = 3e-4
	with mirrored_strategy.scope():
		model        = DCGAN()
		model_gopt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
		model_copt   = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=0.2, beta_2=0.5)
		ckpt         = tf.train.Checkpoint(step=tf.Variable(1), model=model, gopt=model_gopt, copt=model_copt)
		ckpt_manager = tf.train.CheckpointManager(ckpt, directory='experiments/ckpt', max_to_keep=300)
		# ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()

	print(ckpt.step.numpy())
	# START         = int(ckpt.step.numpy()) // len(train_batch) + 1
	START = 0
	EPOCHS        = 300#670#66
	model_freq    = 355#178#355#178#200#40
	t_visfreq     = 355#178#355#178#200#1500#40
	latent        = tf.random.uniform(shape=(16, latent_dim), minval=-0.2, maxval=0.2)
	latent        = tf.concat([latent, latent_Y[:16]], axis=-1)
	print(latent_Y.shape, latent.shape)

	if ckpt_manager.latest_checkpoint:
		print('Restored from last checkpoint epoch: {0}'.format(START))

	if not os.path.isdir('experiments/results'):
		os.makedirs('experiments/results')

	with tf.device('/GPU:0'):
		for epoch in range(START, EPOCHS):
			t_gloss = tf.keras.metrics.Mean()
			t_closs = tf.keras.metrics.Mean()

			tq = tqdm(train_batch)
			for idx, (E, Y, X) in enumerate(tq, start=1):
				batch_size   = X.shape[0]
				_, C = triplenet(E, training=False)
				gloss, closs = dist_train_step(mirrored_strategy, model, model_gopt, model_copt, X, C, latent_dim, batch_size)
				gloss = tf.reduce_mean(gloss)
				closs = tf.reduce_mean(closs)
				t_gloss.update_state(gloss)
				t_closs.update_state(closs)
				ckpt.step.assign_add(1)
				if (idx%model_freq)==0:
					ckpt_manager.save()
				if (idx%t_visfreq)==0:
					# latent_c = tf.concat([latent, C[:16]], axis=-1)
					X = mirrored_strategy.run(model.gen, args=(latent,))
					# X = X.values[0]
					print(X.shape, latent_label.shape)
					show_batch_images(X, save_path='experiments/results/{}.png'.format(int(ckpt.step.numpy())), Y=latent_label)

				tq.set_description('E: {}, gl: {:0.3f}, cl: {:0.3f}'.format(epoch, t_gloss.result(), t_closs.result()))
				# break

			with open('experiments/log.txt', 'a') as file:
				file.write('Epoch: {0}\tT_gloss: {1}\tT_closs: {2}\n'.format(epoch, t_gloss.result(), t_closs.result()))
			print('Epoch: {0}\tT_gloss: {1}\tT_closs: {2}'.format(epoch, t_gloss.result(), t_closs.result()))


			if (epoch%10)==0:
				save_path = 'experiments/inception/{}'.format(epoch)

				if not os.path.isdir(save_path):
					os.makedirs(save_path)

				for cl in range(n_classes):
					test_noise  = np.random.uniform(size=(test_eeg_cls[cl].shape[0],128), low=-1, high=1)
					noise_lst   = np.concatenate([test_noise, test_eeg_cls[cl]], axis=-1)

					for idx, noise in enumerate(tqdm(noise_lst)):
						X = mirrored_strategy.run(model.gen, args=(tf.expand_dims(noise, axis=0),))
						X = cv2.cvtColor(tf.squeeze(X).numpy(), cv2.COLOR_RGB2BGR)
						X = np.uint8(np.clip((X*0.5 + 0.5)*255.0, 0, 255))
						cv2.imwrite(save_path+'/{}_{}.jpg'.format(cl, idx), X)

				# eeg_feature_vectors_test = np.array([test_eeg_features[np.random.choice(np.where(test_eeg_y == test_label)[0], size=(1,))[0]] for test_label in test_labels])
				# latent_var  = np.concatenate([test_noise, eeg_feature_vectors_test], axis=-1)
				# print(test_noise.shape, eeg_feature_vectors_test.shape, latent_var.shape)
				# for idx, noise in enumerate(tqdm(latent_var)):
				# 	X = mirrored_strategy.run(model.gen, args=(tf.expand_dims(noise, axis=0),))
				# 	X = cv2.cvtColor(tf.squeeze(X).numpy(), cv2.COLOR_RGB2BGR)
				# 	X = np.uint8(np.clip((X*0.5 + 0.5)*255.0, 0, 255))
				# 	cv2.imwrite(save_path+'/{}_{}.jpg'.format(test_labels[idx], idx), X)
				# print(X.shape)
			# break
