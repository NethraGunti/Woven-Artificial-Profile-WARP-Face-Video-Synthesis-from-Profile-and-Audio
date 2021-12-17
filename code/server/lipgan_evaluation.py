# -*- coding: utf-8 -*-
"""LipGAN_Evaluation.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1fDTbMc0i4JoO2DCjfN8H97dgdHFZ03My
"""

# !rm -Rf sample_data

base_dir = './'

logs_dir = './'
checkpoint_name = 'residual_model.h5'
resume_model = logs_dir+checkpoint_name
face_det_checkpoint = logs_dir+'mmod_human_face_detector.dat'

face_input_folder = 'C:/Users/Admin/Desktop/btp/app/public/fakefaces/'
voice_input_folder = 'C:/Users/Admin/Desktop/btp/app/public/voices/'
output_folder = 'C:/Users/Admin/Desktop/btp/dump/'

img_file = face_input_folder+'sample_img0.png'
audio_file = voice_input_folder+'sample_audio0.wav'
output_name = 'result0'

face_det_batch_size = 64
lipgan_batch_size = 256
img_size = 96
pads = [0, 0, 0, 0]
fps = 25
mel_step_size = 27
mel_idx_multiplier = 80./fps

# !cp drive/MyDrive/BTP21SRD01/LipGAN/Preprocessing/*.py .

from os import listdir, path
import numpy as np
import scipy, cv2, os, sys, librosa, audio
import dlib, json, h5py, subprocess
from tqdm import tqdm
import shutil
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2DTranspose, Conv2D, BatchNormalization,\
                        Activation, Concatenate, Input, MaxPool2D,\
						UpSampling2D, ZeroPadding2D, Lambda, Add
from tensorflow.keras.callbacks import ModelCheckpoint, Callback
from tensorflow.keras import backend as K
from tensorflow.keras.utils import plot_model

"""### Functions to Create Model"""

def conv_block(x, num_filters, kernel_size=3, strides=1, padding='same', act=True):
    """
    function to add conv blocks
    Args:
        x -> tensor/ndarry: feature vector
        num_filters -> int: number of filters to be applied
        kernel_size -> int: filter size = (kernel_size x kernel_size) (default: 3)
        strides -> int: default 1
        padding-> str: 'same' to generate output with dimensions same as input
        act -> boolean: whether to use activation function or not
    """

    x = Conv2D(filters=num_filters, kernel_size= kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(momentum=.8)(x)
    if act:
        x = Activation('relu')(x)
    return x

def conv_t_block(x, num_filters, kernel_size=3, strides=2, padding='same'):
    """
    function to add conv transpose blocks
    Args:
        x -> tensor/ndarry: feature vector
        num_filters -> int: number of filters to be applied
        kernel_size -> int: filter size = (kernel_size x kernel_size) (default: 3)
        strides -> int: default 1
        padding-> str: 'same' to generate output with dimensions same as input
        act -> boolean: whether to use activation function or not
    """

    x = Conv2DTranspose(filters=num_filters, kernel_size= kernel_size, strides=strides, padding=padding)(x)
    x = BatchNormalization(momentum=.8)(x)
    x = Activation('relu')(x)
    return x

def residual_block(inp, num_filters):
    x = conv_block(inp, num_filters)
    x = conv_block(x, num_filters)
    x = Add()([x, inp])
    x = Activation('relu') (x)
    return x

def create_model(img_size=img_size, mel_step_size=mel_step_size):
    """
    function to create the model
    Parts:
        face encoder, audio encoder, decoder
    Args:
        img_size -> input for face encoder: (img_size x img_size)
        mel_step_size -> input for audio encoder: (80, mel_step_size)
    """

    ### face encoder
    input_face = Input(shape=(img_size, img_size, 6), name="input_face")

    identity_mapping = conv_block(input_face, 32, kernel_size=7) # 96x96

    x1_face = conv_block(identity_mapping, 64, kernel_size=5, strides=2) # 48x48
    x1_face = residual_block(x1_face, 64)
    x1_face = residual_block(x1_face, 64)

    x2_face = conv_block(x1_face, 128, 3, 2) # 24x24
    x2_face = residual_block(x2_face, 128)
    x2_face = residual_block(x2_face, 128)
    x2_face = residual_block(x2_face, 128)

    x3_face = conv_block(x2_face, 256, 3, 2) #12x12
    x3_face = residual_block(x3_face, 256)
    x3_face = residual_block(x3_face, 256)

    x4_face = conv_block(x3_face, 512, 3, 2) #6x6
    x4_face = residual_block(x4_face, 512)
    x4_face = residual_block(x4_face, 512)

    x5_face = conv_block(x4_face, 512, 3, 2) #3x3
    x6_face = conv_block(x5_face, 512, 3, 1, padding='valid')
    x7_face = conv_block(x6_face, 512, 1, 1)

    ### audio encoder
    input_audio = Input(shape=(80, mel_step_size, 1), name="input_audio")

    x = conv_block(input_audio, 32)
    x = residual_block(x, 32)
    x = residual_block(x, 32)

    x = conv_block(x, 64, strides=3)	#27X9
    x = residual_block(x, 64)
    x = residual_block(x, 64)

    x = conv_block(x, 128, strides=(3, 1)) 		#9X9
    x = residual_block(x, 128)
    x = residual_block(x, 128)

    x = conv_block(x, 256, strides=3)	#3X3
    x = residual_block(x, 256)
    x = residual_block(x, 256)

    x = conv_block(x, 512, strides=1, padding='valid')	#1X1
    x = conv_block(x, 512, 1, 1)

    embedding = Concatenate(axis=3)([x7_face, x])

    ### deocder
    x = conv_t_block(embedding, 512, 3, 3)# 3x3
    x = Concatenate(axis=3) ([x5_face, x]) 

    x = conv_t_block(x, 512) #6x6
    x = residual_block(x, 512)
    x = residual_block(x, 512)
    x = Concatenate(axis=3) ([x4_face, x])

    x = conv_t_block(x, 256) #12x12
    x = residual_block(x, 256)
    x = residual_block(x, 256)
    x = Concatenate(axis=3) ([x3_face, x])

    x = conv_t_block(x, 128) #24x24
    x = residual_block(x, 128)
    x = residual_block(x, 128)
    x = Concatenate(axis=3) ([x2_face, x])

    x = conv_t_block(x, 64) #48x48
    x = residual_block(x, 64)
    x = residual_block(x, 64)
    x = Concatenate(axis=3) ([x1_face, x])

    x = conv_t_block(x, 32) #96x96
    x = Concatenate(axis=3) ([identity_mapping, x])
    x = conv_block(x, 16) #96x96
    x = conv_block(x, 16) #96x96

    x = Conv2D(filters=3, kernel_size=1, strides=1, padding="same") (x)
    prediction = Activation("sigmoid", name="prediction")(x)

    model = Model(inputs=[input_face, input_audio], outputs=prediction)
    return model

"""### Functions to Generate Data"""

### functions for preprocessing the input (same as from the preprocessing stage)

def rect_to_bb(d):
	"""
	args: 'valid' face frame
	returns: bounding box coords
	"""
	x = d.rect.left()
	y = d.rect.top()
	w = d.rect.right() - x
	h = d.rect.bottom() - y
	return (x, y, w, h)

def calcMaxArea(rects):
	"""
	args: takes a list of 'valid' frames
	returns: frame with the max bb are
	"""
	max_cords = (-1,-1,-1,-1)
	max_area = 0
	max_rect = None
	for i in range(len(rects)):
		cur_rect = rects[i]
		(x,y,w,h) = rect_to_bb(cur_rect)
		if w*h > max_area:
			max_area = w*h
			max_cords = (x,y,w,h)
			max_rect = cur_rect
	return max_cords, max_rect

def face_detect(images):
	detector = dlib.cnn_face_detection_model_v1(face_det_checkpoint)
	batch_size = face_det_batch_size

	predictions = []
	for i in tqdm(range(0, len(images), batch_size)):
		predictions.extend(detector(images[i:i + batch_size]))
	
	results = []
	pady1, pady2, padx1, padx2 = pads
	for rects, image in zip(predictions, images):
		(x, y, w, h), max_rect = calcMaxArea(rects)
		if x == -1:
			results.append([None, (-1,-1,-1,-1), False])
			continue
		y1 = max(0, y + pady1)
		y2 = min(image.shape[0], y + h + pady2)
		x1 = max(0, x + padx1)
		x2 = min(image.shape[1], x + w + padx2)
		face = image[y1:y2, x1:x2, ::-1] # RGB ---> BGR

		results.append([face, (y1, y2, x1, x2), True])
	
	del detector
	return results

def datagen(frames, mels):
	"""
    func: data generator to generate frames from mel spec
    args:
        frames -> list of frames // in this case a list containing input image
        mels -> list of mel_chunks
    returns: object of type generator: (image batch, melspec batch, frame batch, coords batch)
	"""
	img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []
	face_det_results = face_detect([frames[0][...,::-1]])           # list of detected frontal face views/coords

    ### detect valid frames and initialize batches
	for i, m in enumerate(mels):
		idx = 0
		frame_to_save = frames[idx].copy()
		face, coords, valid_frame = face_det_results[idx].copy()
		if not valid_frame:
			print ("Face not detected, skipping frame {}".format(i))
			continue

		face = cv2.resize(face, (img_size, img_size))

		img_batch.append(face)
		mel_batch.append(m)
		frame_batch.append(frame_to_save)
		coords_batch.append(coords)

		### add masks to inputs after creating several batches
		img_batch, mel_batch = np.asarray(img_batch), np.asarray(mel_batch)

		img_masked = img_batch.copy()
		img_masked[:, img_size//2:] = 0

		img_batch = np.concatenate((img_batch, img_masked), axis=3) / 255.
		mel_batch = np.reshape(mel_batch, [len(mel_batch), mel_batch.shape[1], mel_batch.shape[2], 1])

		yield img_batch, mel_batch, frame_batch, coords_batch
		img_batch, mel_batch, frame_batch, coords_batch = [], [], [], []

"""### Main"""

def get_vid(face=img_file, audio_file=audio_file, output_name=output_name, curr_count=0):
	"""
    func: generate and save video in `output_folder`
    args:
        face -> path to the image file
        audio_file -> path to the audio file
	"""
	full_frames = [cv2.resize(cv2.imread(face), (1024, 1024))]            # read face image
	wav = audio.load_wav(audio_file, 16000)     # load the wav file
	mel = audio.melspectrogram(wav)             # generate the melspec
	# print(len(full_frames), mel.shape)
	if np.isnan(mel.reshape(-1)).sum() > 0:
		raise ValueError('Mel contains nan!')
	
    ### break the melspec for each frame
	mel_chunks = []
	i = 0
	while True:
		start_idx = int(i * mel_idx_multiplier)
		if start_idx + mel_step_size > len(mel[0]):
			break
		mel_chunks.append(mel[:, start_idx : start_idx + mel_step_size])
		i += 1
	# print("Length of mel chunks: {}".format(len(mel_chunks)))       # length = 305
	batch_size = lipgan_batch_size
	gen = datagen(full_frames.copy(), mel_chunks)
 
    ### initialise a video (audioless) using opencv for the generated data
	for i, (img_batch, mel_batch, frames, coords) in enumerate(tqdm(gen,total=int(np.ceil(float(len(mel_chunks))/batch_size)))):
		if i == 0:
			model = create_model()
			# print ("Model Created")

			model.load_weights(resume_model)
			# print ("Model loaded")
			
			frame_h, frame_w = full_frames[0].shape[:-1]
			out = cv2.VideoWriter(path.join(output_folder, f'frames{curr_count}.mp4'), cv2.VideoWriter_fourcc(*'MP4V'), fps, (frame_w, frame_h))

        ### predict the frames and write them in the video
		pred = model.predict([img_batch, mel_batch])
		pred = pred * 255
		
		for p, f, c in zip(pred, frames, coords):
			y1, y2, x1, x2 = c
			p = cv2.resize(p, (x2 - x1, y2 - y1))

			f[y1:y2, x1:x2] = p
			out.write(f)

	out.release()

    ### combine the audio and video to generate a final video (with audio)
	command = 'ffmpeg -i {} -i {} -strict -2 -q:v 1 {}'.format(audio_file, path.join(output_folder, f'frames{curr_count}.mp4'), path.join(output_folder, f'{output_name}.mp4'))
	subprocess.call(command, shell=True)
	# original = f"C:/Users/Admin/Desktop/btp/{output_name}.mp4"
	# target = f"C:/Users/Admin/Desktop/btp/app/public/vid/{output_name}.mp4"
	# shutil.move(original, target)

# get_vid()

