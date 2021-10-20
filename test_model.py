import numpy as np
import skipthoughts
import utils
import tensorflow as tf
import tensorlayer as tl

from models_resnet import generator_txt2img_resnet

Generator = generator_txt2img_resnet([None, 356], is_train=False)

tl.files.load_hdf5_to_weights('Output/checkpoint/G.h5', Generator)
Generator.eval()

test_descr = ['forest near glacier mountain during day']

embedded_model = skipthoughts.load_model()
print('Creation of skipthought vectors : loading ....')

caption_vectors = skipthoughts.encode(embedded_model, test_descr)
tensor_text_embedding = tf.convert_to_tensor(caption_vectors)
reduced_text_embedding = utils.lrelu(utils.linear(tensor_text_embedding, 256))

z = np.random.normal(loc=0.0, scale=1.0, size=[1, 100]).astype(np.float32)
z = tf.convert_to_tensor(z)

z_text_concat = tf.concat([z, reduced_text_embedding], 1)
img = Generator(z_text_concat)

img = img.numpy().squeeze().astype(np.uint8)
tl.visualize.save_image(img, 'model_test/img_test.jpg')