This project aim to train a GAN model to generate landscape image from a textual description.
It is inspired by the work of https://github.com/paarthneekhara/text-to-image

The following is the model architecture. The blue bars represent the Skip Thought Vectors for the captions.

![Model architecture](http://i.imgur.com/dNl2HkZ.jpg)

Image Source : [Generative Adversarial Text-to-Image Synthesis][2] Paper

## Requirements
- Python 2.7.6
- [Tensorflow][4]
- [h5py][5]
- [scikit-learn][7] 
- [NLTK][8]

## Datasets
- The model is currently trained on a home-made dataset with pictures and caption from a website. 

## Usage
- <b>Dataset</b>
  * Scrap image's url and caption with unsplash_webscrapping.py
  * Download image with load_image.py
      
- <b>Generating Images and Captions vectorization</b>
  * Write the captions in text file, and save it as ```Data/sample_captions.txt```. Generate the skip thought vectors for these captions using:
  ```
  python generate_thought_vectors.py --caption_file="Data/sample_captions.txt"
  ```
  * Generate the Images for the thought vectors using:
  ```
  python process_images.py
  ```
  
- <b>Training</b>
  * Basic usage `python train.py`


## References
- [Generative Adversarial Text-to-Image Synthesis][2] Paper
- [Generative Adversarial Text-to-Image Synthesis][11] Code
- [Skip Thought Vectors][1] Paper
- [Skip Thought Vectors][12] Code
- [DCGAN in Tensorflow][3]
- [DCGAN in Tensorlayer][15]

## Alternate Implementations
- [Text to Image in Torch by Scot Reed][11]
- [Text to Image in Tensorlayer by Dong Hao][16]

## License
MIT


[1]:http://arxiv.org/abs/1506.06726
[2]:http://arxiv.org/abs/1605.05396
[3]:https://github.com/carpedm20/DCGAN-tensorflow
[4]:https://github.com/tensorflow/tensorflow
[5]:http://www.h5py.org/
[6]:https://github.com/Theano/Theano
[7]:http://scikit-learn.org/stable/index.html
[8]:http://www.nltk.org/
[9]:http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
[10]:https://drive.google.com/file/d/0B0ywwgffWnLLcms2WWJQRFNSWXM/view
[11]:https://github.com/reedscot/icml2016
[12]:https://github.com/ryankiros/skip-thoughts
[13]:https://github.com/ryankiros/skip-thoughts#getting-started
[14]:https://bitbucket.org/paarth_neekhara/texttomimagemodel/raw/74a4bbaeee26fe31e148a54c4f495694680e2c31/latest_model_flowers_temp.ckpt
[15]:https://github.com/zsdonghao/dcgan
[16]:https://github.com/zsdonghao/text-to-image
