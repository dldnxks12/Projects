'''

Image의 Statistic을 추출하여 Latent Space에서 가감하는 방식으로 해당 Style을 조정할 수 있다.

'''


import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Conv2D, UpSampling2D, Layer
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.models import save_model, load_model
import cv2
from tqdm import tqdm

CONTENT_PATH = '/content/drive/MyDrive/Colab Notebooks/human_segmentation/content/img2.png'
STYLE_PATH = '/content/drive/MyDrive/Colab Notebooks/human_segmentation/style/style.jpg'
BATCH_SIZE = 10
CHECKPOINT_PATH = "/content/drive/MyDrive/Colab Notebooks/human_segmentation/Weights" 

content_image_generator = ImageDataGenerator()
content_data_generator = content_image_generator.flow_from_directory(directory=CONTENT_PATH, batch_size=BATCH_SIZE, shuffle=True)
style_image_generator = ImageDataGenerator()
style_data_generator = style_image_generator.flow_from_directory(directory=STYLE_PATH, batch_size=BATCH_SIZE, shuffle=True)

def preprocess(x):
    # RGB to BGR
    img = tf.reverse(x, axis=[-1])
    img -= np.array([103.939, 116.779, 123.68])
    return img
def deprocess(x):
    # BGR to RGB
    img = x + np.array([103.939, 116.779, 123.68])
    img = tf.reverse(img, axis=[-1])
    img = tf.clip_by_value(img, 0.0, 255.0)
    return img
def get_one_mixed_batch():
    contentX, _ = next(content_data_generator)
    styleX, _ = next(style_data_generator)
    return contentX, styleX
    
def save_weights(model, epoch, path = CHECKPOINT_PATH):
    model.save_weights(path + '/weights{}.h5'.format(epoch))
    print('Saved weights for epoch {}'.format(epoch))
    
class AdaIN(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        self.eps = epsilon
        super(AdaIN, self).__init__(**kwargs)

    def call(self, inputs):
        content, style, alpha = inputs

        meanC, varC = tf.nn.moments(content, [1, 2], keepdims=True)
        meanS, varS = tf.nn.moments(style, [1, 2], keepdims=True)

        sigmaC = tf.sqrt(tf.add(varC, self.eps))
        sigmaS = tf.sqrt(tf.add(varS, self.eps))

        adain = (content - meanC) * sigmaS / sigmaC + meanS
        return alpha * adain + (1 - alpha) * content

    def get_config(self):
        config = super().get_config().copy()
        return config

class ReflectionPadding2D(tf.keras.layers.Layer):

    def __init__(self, padding, **kwargs):
        super(ReflectionPadding2D, self).__init__(**kwargs)
        self.paddings = ((0, 0), (padding, padding), (padding, padding), (0, 0))

    def call(self, x):
        return tf.pad(x, paddings=self.paddings, mode='REFLECT')

    def get_config(self):
        config = {'paddings': self.paddings}
        base_config = super(ReflectionPadding2D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))        

def build_vgg19():
    vgg = tf.keras.applications.VGG19(include_top=False, weights='imagenet')
    vgg.trainable = False
    outputs = vgg.get_layer('block4_conv1').output
    model = tf.keras.Model([vgg.input], outputs, name='VGG_Encoder')
    return model
def build_vgg19_relus(vgg19):
    relus = ['block1_conv1', 'block2_conv1', 'block3_conv1', 'block4_conv1']
    features = [vgg19.get_layer(relu).output for relu in relus]
    vgg19_relus = Model(inputs=vgg19.input, outputs=features)
    vgg19_relus.trainable = False
    return vgg19_relus

def build_decoder(input_shape):
    input = Input(shape = input_shape)
    x = input
    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(256, (3,3), activation='relu', padding='valid')(x)
    x = UpSampling2D((2,2))(x)

    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(256, (3,3), activation='relu', padding='valid')(x)
    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(256, (3,3), activation='relu', padding='valid')(x)
    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(256, (3,3), activation='relu', padding='valid')(x)
    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(128, (3,3), activation='relu', padding='valid')(x)
    x = UpSampling2D((2,2))(x)

    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(128, (3,3), activation='relu', padding='valid')(x)
    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(64, (3,3), activation='relu', padding='valid')(x)
    x = UpSampling2D((2,2))(x)

    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(64, (3,3), activation='relu', padding='valid')(x)
    x = ReflectionPadding2D(padding=1)(x)
    x = Conv2D(3, (3,3), padding='valid')(x)
    output = x
    return Model(inputs=[input],outputs=[output])    

def build_model(encoder, decoder, input_shape):
    content = Input(shape=input_shape, name='content')
    style = Input(shape=input_shape, name = 'style')
    alpha = Input(shape=(1,), name='alpha')

    enc_content = encoder(content)
    enc_style = encoder(style)

    adain = AdaIN()([enc_content, enc_style, alpha])

    out = decoder(adain)

    return Model(inputs=[content, style, alpha], outputs=[out, adain])  

def get_loss(encoder, vgg19_relus, epsilon=1e-5, style_weight=10):
    def loss(y_true, y_pred):
        # y_true == input == [content, style]
        out, adain = y_pred[0], y_pred[1]

        # Encode output and compute content_loss
        out = deprocess(out)
        out = preprocess(out)
        enc_out = encoder(out)
        content_loss = tf.reduce_sum(tf.reduce_mean(tf.square(enc_out - adain), axis=[1, 2]))

        # Compute style loss from vgg relus
        style = y_true[1]
        style_featues = vgg19_relus(style)
        gen_features = vgg19_relus(out)
        style_layer_loss = []
        for enc_style_feat, enc_gen_feat in zip(style_featues, gen_features):
            meanS, varS = tf.nn.moments(enc_style_feat, [1, 2])
            meanG, varG = tf.nn.moments(enc_gen_feat, [1, 2])

            sigmaS = tf.sqrt(varS + epsilon)
            sigmaG = tf.sqrt(varG + epsilon)

            l2_mean = tf.reduce_sum(tf.square(meanG - meanS))
            l2_sigma = tf.reduce_sum(tf.square(sigmaG - sigmaS))

            style_layer_loss.append(l2_mean + l2_sigma)

        style_loss = tf.reduce_sum(style_layer_loss)

        # Compute the total loss
        weighted_style_loss = style_weight * style_loss
        total_loss = content_loss + weighted_style_loss
        return total_loss, content_loss, weighted_style_loss

    return loss    

def train(model, loss, n_epochs=10):
    optimizer = Adam(lr=1e-4, decay=5e-5)
    n_batches = len(style_data_generator)
    alpha = 1.0
    for e in range(1, n_epochs + 1):
        losses = {"total": 0.0, "content": 0.0, "style": 0.0}

        pbar = tqdm(total=n_batches, ncols=50)
        for i in range(n_batches):
            # Get batch
            content, style = get_one_mixed_batch()
            content = preprocess(content)
            style = preprocess(style)
            if content is None or style is None:
                break
            # Train on batch
            # total_loss, content_loss, weighted_style_loss, weighted_color_loss
            with tf.GradientTape() as tape:
                prediction = model([content, style, alpha])
                loss_values = loss([content, style], prediction)
            grads = tape.gradient(loss_values[0], model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))
            pbar.update(1)
        save_weights(model,e)    

def get_image(img_path, resize=True, shape=(256,256)):
    image = cv2.imread(img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    if resize:
        image = cv2.resize(image, shape)
    image = image.astype(np.float32)
    return image

INPUT_SHAPE = (None, None, 3)
EPOCHS = 10
EPSILON = 1e-5

# content_batch, style_batch = get_one_mixed_batch()



vgg19 = build_vgg19()  # encoder
decoder = build_decoder(vgg19.output.shape[1:])  # input shape == encoder output shape
model = build_model(vgg19, decoder, INPUT_SHAPE)

model.load_weights(CHECKPOINT_PATH+'/weights6.h5')

vgg19_relus = build_vgg19_relus(vgg19)
loss = get_loss(vgg19, vgg19_relus, epsilon=EPSILON)

# train(model, loss, n_epochs=EPOCHS)    

INPUT_SHAPE = (None, None, 3)
EPOCHS = 10
EPSILON = 1e-5

vgg19 = build_vgg19()  # encoder
decoder = build_decoder(vgg19.output.shape[1:])  # input shape == encoder output shape
model = build_model(vgg19, decoder, INPUT_SHAPE)
model.load_weights(CHECKPOINT_PATH+'/weights6.h5')
CONTENT_PATH = CONTENT_PATH
STYLE_PATH = STYLE_PATH

content = get_image(CONTENT_PATH, resize=False)
style = get_image(STYLE_PATH, resize=False)

content = preprocess(content)
style = preprocess(style)
content = content[np.newaxis]
style = style[np.newaxis]
alpha = 0.2
pred = model([content,style,alpha])[0]
pred = tf.squeeze(pred)
pred = deprocess(pred)

plt.imshow(pred)

pred = pred/255

plt.figure(figsize = (16,16))
plt.imshow(pred)

import imageio

pred = np.array(pred)

imageio.imwrite('/content/drive/MyDrive/Colab Notebooks/human_segmentation/content/img2_result.png', pred)

pred = np.array(pred, dtype = np.uint8)

imageio.imwrite('/content/drive/MyDrive/Colab Notebooks/human_segmentation/content/img2_result2.png', pred)    
