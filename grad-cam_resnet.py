from keras.applications.vgg16 import VGG16
from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input
from keras.applications.imagenet_utils import decode_predictions
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
  #  print "loss_shape:{}".format(input_shape)
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

  #  print model.layers
    target_layer = lambda x: target_category_loss(x, category_index, nb_classes)
    # Added the layer which gets the loss for the predicted category
    model.add(Lambda(target_layer,
                     output_shape=target_category_loss_output_shape))
    # get the loss for the predicted category
    # loss means output of that model for the given category
    # i.e. output when setting the predicated category as only category

    loss = K.sum(model.layers[-1].output)
    print "loss: {}".format(loss)
    # get the convolution layers output

    conv_output = [l for l in model.layers[0].layers if l.name == layer_name][0].output
    print "conv_output: {}".format(conv_output)
   # print model.layers[0].get_layer[layer_name].name, model.layers[0].get_layer[layer_name].output_shape

    # get the gradient with the respect to the predicted category loss and convolution outputs
    # and normalized them also
    # TODO: apply Relu on convolution maps
    grads = normalize(K.gradients(loss, conv_output)[0])
    print "grads: {}".format(grads)
    # made a function which givens grads and convolution outputs
    gradient_function = K.function([model.layers[0].input,  K.learning_phase()], [conv_output, grads])
    # pass the image thought it
    output, grads_val = gradient_function([image, 0])
    output, grads_val = output[0, :], grads_val[0, :, :, :]
    print "output: {} grad_val: {}".format(output.shape, grads_val.shape)
    # get weights of each convolution layers as mean of GAP of the gradients
    weights = np.mean(grads_val, axis=(0, 1))
    maps_weights = np.mean(output, axis=(0, 1))
    print 'mmax: {}, mmin: {}, mavg: {} sum: {}'.format(np.max(maps_weights, axis=0),
                                                        np.min(maps_weights, axis=0), np.mean(maps_weights, axis=0),
                                                        np.sum(maps_weights, axis=0))

    print weights.shape
    #   print weights
    print 'wmax: {}, wmin: {}, wavg: {} sum: {}'.format(np.max(weights, axis=0),
                                                        np.min(weights, axis=0), np.mean(weights, axis=0),
                                                        np.sum(weights, axis=0))
     # prepare heatmap
    cam = np.ones(output.shape[0:2], dtype=np.float32)
    # Add weighted activation maps
    for i, w in enumerate(weights):
        # Added relu
        print i, w, maps_weights[i]
        temp = (w*output[:, :, i])
        np.maximum(temp, 0, temp)
        cam += temp
   # np.maximum(cam,0,cam)
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
    layer_name = 'activation_49'
    model = ResNet50(include_top=True, weights='imagenet')
   # model.summary()
   # print "res5c_branch2c.shape: {}".format(model.get_layer('res5c_branch2c').output_shape)
    preds = model.predict(preprocessed_input)
    print 'Predicted:', decode_predictions(preds, top=3)
    predicted_class = np.argmax(preds)
    cam = grad_cam(model, preprocessed_input, predicted_class, layer_name)
    save_name = sys.argv[1][0:-4] + '_' + decode_predictions(preds, top=1)[0][0][1] + '_ac49_resnet.jpg'
    cv2.imwrite(save_name, cam)
