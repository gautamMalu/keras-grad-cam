from keras.applications.vgg16 import VGG16
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.layers.core import Lambda
from keras.models import Sequential
import keras.backend as K
import tensorflow as tf
import numpy as np
import sys
import cv2

def target_category_loss(x, category_index, nb_classes):
    '''
    loss layer which gives the loss for the predicted category
    :param x:
    :param category_index:
    :param nb_classes:
    :return:
    '''
    return tf.mul(x, K.one_hot([category_index], nb_classes))

def target_category_loss_output_shape(input_shape):
    return input_shape

def normalize(x):
    # utility function to normalize a tensor by its L2 norm
    return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

def load_image(path):
    img_path = sys.argv[1]
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    return x

def grad_cam(input_model, image, category_index, layer_name): 
    model = Sequential()
    model.add(input_model)

    nb_classes = 1000
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    # Added the layer which gets the loss for the predicted category
    model.add(Lambda(target_layer,
                     output_shape=target_category_loss_output_shape))
    # get the loss for the predicted category
    loss = K.sum(model.layers[-1].output)
    # get the convolution layers output
    conv_output =  [l for l in model.layers[0].layers if l.name is layer_name][0].output

    # get the gradient with the respect to the predicted category loss and convolution outputs
    # and normalized them also
    # TODO: apply Relu on convolution maps
    grads = normalize(K.gradients(loss, conv_output)[0])
    # made a function which givens grads and convolution outputs
    gradient_function = K.function([model.layers[0].input], [conv_output, grads])
    # pass the image thought it
    output, grads_val = gradient_function([image])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    # get weights of each convolution layers as mean of GAP of the gradients
    weights = np.mean(grads_val, axis=(0, 1))
     # prepare heatmap
    cam = np.ones(output.shape[0:2], dtype=np.float32)
    # Add weighted activation maps
    for i, w in enumerate(weights):
        # Added relu
        temp = (w*output[:, :, i])
        np.maximum(temp, 0, temp)
        cam += temp
    # resize and normalization
    cam = cv2.resize(cam, (224, 224))
    cam = np.maximum(cam, 0)
    cam = cam / np.max(cam)

    #Return to BGR [0..255] from the preprocessed image
    image = image[0, :]
    image -= np.min(image)
    image = np.minimum(image, 255)

    cam = cv2.applyColorMap(np.uint8(255*cam), cv2.COLORMAP_JET)
    cam = np.float32(cam) + np.float32(image)
    cam = 255 * cam / np.max(cam)
    return np.uint8(cam)

if __name__ == '__main__':
    preprocessed_input = load_image(sys.argv[1])

    model = VGG16(include_top=True, weights='imagenet')

    predicted_class = np.argmax(model.predict(preprocessed_input))
    cam = grad_cam(model, preprocessed_input, predicted_class, "block5_pool")
    save_name = sys.argv[1][0:-4]+'_cam.jpg'
    cv2.imwrite(save_name, cam)
