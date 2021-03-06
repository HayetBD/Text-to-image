import time
import numpy as np
import tensorflow as tf
import tensorlayer as tl
import utils
from models_resnet import generator_txt2img_resnet, discriminator_txt2img_resnet

t_dim = 256
n_epoch = 500 # "Epoch to train [25]"
z_dim = 100 # "Num of noise value]"
lr = 0.0002 # "Learning rate of for adam [0.0002]")
beta1 = 0.5 # "Momentum term of adam [0.5]")
batch_size = 100 # "The number of batch images [64]")
output_size = 64 # "The size of the output images to produce [64]")
sample_size = 64 # "The number of sample images [64]")
c_dim = 3 # "Number of image channels. [3]")
save_every_epoch = 5 # "The interval of saving checkpoints.")

checkpoint_dir = "Output/checkpoint" # "Directory name to save the checkpoints [checkpoint]")
sample_dir = "Output/output" # "Directory name to save the image samples [samples]")
log_dir = "Output/Log_loss"

tl.files.exists_or_mkdir(checkpoint_dir) # save model
tl.files.exists_or_mkdir(sample_dir) # save generated image
tl.files.exists_or_mkdir(log_dir) # save generated image


# get vectorized images and caption
# skipvectors_sample.npy created in generate_thought_vectors.py
array_caption_vector = np.load('Data/vectors_files/skipvectors_sample.npy')
#imgvectors_sample.npy created in process_images.py
array_image_vector = np.load('Data/vectors_files/imgvectors_sample.npy')

tensor_captions = tf.convert_to_tensor(array_caption_vector)

tensor_images = tf.convert_to_tensor(array_image_vector)

def train():
    reduced_text_embedding = utils.lrelu(utils.linear(tensor_captions, 256))
    #G = get_generator([None, z_dim+reduced_text_embedding.shape[1]])
    #D = get_discriminator([None, output_size, output_size, c_dim], input_rnn_embed = [None, reduced_text_embedding.shape[1]])

    G = generator_txt2img_resnet([None, z_dim+reduced_text_embedding.shape[1]])
    #tl.files.load_hdf5_to_weights(checkpoint_dir+'/G.h5', G)
    D = discriminator_txt2img_resnet([None, output_size, output_size, c_dim],t_txt=[None, reduced_text_embedding.shape[1]])
    #tl.files.load_hdf5_to_weights(checkpoint_dir+'/D.h5', D)

    G.train()
    D.train()

    d_optimizer = tf.optimizers.Adam(lr, beta_1=beta1)
    g_optimizer = tf.optimizers.Adam(lr, beta_1=beta1)

    n_step_epoch = int(len(tensor_images) // batch_size)
    dataset_img = tf.data.Dataset.from_tensor_slices(tensor_images)
    batched_ds_img = dataset_img.batch(batch_size, drop_remainder=True)

    dataset_txt = tf.data.Dataset.from_tensor_slices(reduced_text_embedding)
    batched_ds_txt = dataset_txt.batch(batch_size, drop_remainder=True)
    list_batch_txt = [batch for batch in batched_ds_txt]

    text_file = open(log_dir+"/log.txt", "a")
    text_file.write('epoch, d_loss, g_loss\n')
    text_file.close()

    for epoch in range(n_epoch):
        for step, batch_images in enumerate(batched_ds_img):
            batch_txt = list_batch_txt[step]
            step_time = time.time()
            with tf.GradientTape(persistent=True) as tape:
                z = np.random.normal(loc=0.0, scale=1.0, size=[batch_size, z_dim]).astype(np.float32)
                z = tf.convert_to_tensor(z)
                z_text_concat = tf.concat([z, batch_txt], 1)
                net_fake_image = G(z_text_concat)

                cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=False)
                disc_fake_image_logits = D([net_fake_image, batch_txt])
                g_loss = cross_entropy(tf.ones_like(disc_fake_image_logits),disc_fake_image_logits)

                disc_real_image_logits = D([batch_images, batch_txt])
                d_loss1 = cross_entropy(tf.ones_like(disc_real_image_logits),disc_real_image_logits)

                idexs = utils.get_random_int(min=0, max=len(batch_txt) - 1, number=batch_size)
                wrong_caption = [batch_txt[i]for i in idexs]
                wrong_caption = tf.stack(wrong_caption)
                disc_mismatch_logits = D([batch_images, wrong_caption])
                d_loss2 = cross_entropy(tf.zeros_like(disc_mismatch_logits), disc_mismatch_logits)
                d_loss3 = cross_entropy(tf.zeros_like(disc_fake_image_logits), disc_fake_image_logits)
                d_loss = d_loss1 + (d_loss2 + d_loss3) * 0.5

            grad = tape.gradient(g_loss, G.trainable_weights)
            g_optimizer.apply_gradients(zip(grad, G.trainable_weights))
            if(step%2 == 0):
                grad = tape.gradient(d_loss, D.trainable_weights)
                d_optimizer.apply_gradients(zip(grad, D.trainable_weights))
            del tape

            print("Epoch: [{}/{}] [{}/{}] took: {:.3f}, d_loss: {:.5f}, g_loss: {:.5f}".format(epoch, \
                                                                                               n_epoch, step,
                                                                                               n_step_epoch,
                                                                                               time.time() - step_time,
                                                                                               d_loss, g_loss))
            text_file = open(log_dir+"/log.txt", "a")
            text_file.write('{}, {:.5f}, {:.5f} \n'.format(epoch, d_loss, g_loss))
            text_file.close()

        if np.mod(epoch, save_every_epoch) == 0:
            #G.save_weights('{}/G_weights.npz'.format(checkpoint_dir), format='npz')
            #D.save_weights('{}/D_weights.npz'.format(checkpoint_dir), format='npz')
            G.save('{}/G.h5'.format(checkpoint_dir), save_weights=True)
            G.eval()
            D.save('{}/D.h5'.format(checkpoint_dir), save_weights=True)
            D.eval()

            z = np.random.normal(loc=0.0, scale=1.0, size=[4, z_dim]).astype(np.float32)
            z = tf.convert_to_tensor(z)

            test_descr = np.reshape(batch_txt[0:4], (4,256))
            z_text_concat = tf.concat([z, test_descr], 1)

            result = G(z_text_concat)

            G.train()
            D.train()

            img = result.numpy().squeeze().astype(np.uint8)

            #Sauvegarde plusieurs images
            tl.visualize.save_images(img, [2, 2], '{}/train_{:02d}.png'.format(sample_dir, epoch))

            #Sauvegarde une image
            #tl.visualize.save_image(img, '{}/train_{:02d}.png'.format(sample_dir, epoch))


if __name__ == '__main__':
    train()