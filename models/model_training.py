import os
import cv2
import numpy as np

def load_data(image_train_folder, label_train_folder, image_val_folder, label_val_folder, image_test_folder, image_size):
    # Read and preprocess training data
    train_images, train_labels = read_and_preprocess_data(image_train_folder, label_train_folder, image_size)

    # Read and preprocess validation data
    val_images, val_labels = read_and_preprocess_data(image_val_folder, label_val_folder, image_size)

    # Read and preprocess test data
    test_images = read_and_preprocess_test_data(image_test_folder, image_size)

    return train_images, train_labels, val_images, val_labels, test_images


def read_and_preprocess_data(image_folder, label_folder, image_size):
    image_filenames = sorted(os.listdir(image_folder))
    label_filenames = sorted(os.listdir(label_folder))

    images = []
    labels = []

    for image_filename in image_filenames:
        image_path = os.path.join(image_folder, image_filename)
        label_filename = image_filename.replace(".tif", ".tif")  # Adjust the label file extension if needed
        label_path = os.path.join(label_folder, label_filename)

        # Read and resize the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)

        # Normalize the image
        image = image / 255.0

        # Read and resize the label
        label = cv2.imread(label_path, 0)  # Read as grayscale
        label = cv2.resize(label, image_size)

        # Normalize the label
        label = label / 255.0

        # Append to the data lists
        images.append(image)
        labels.append(label)

    # Convert data lists to NumPy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels


def read_and_preprocess_test_data(image_folder, image_size):
    image_filenames = sorted(os.listdir(image_folder))

    images = []

    for image_filename in image_filenames:
        image_path = os.path.join(image_folder, image_filename)

        # Read and resize the image
        image = cv2.imread(image_path)
        image = cv2.resize(image, image_size)

        # Normalize the image
        image = image / 255.0

        # Append to the data list
        images.append(image)

    # Convert data list to a NumPy array
    images = np.array(images)

    return images


# Set your data paths
image_train_folder = "../data/data_train/images"
label_train_folder = "../data/data_train/gt"
image_val_folder = "../data/data_train/images_val"
label_val_folder = "../data/data_train/gt_val"
image_test_folder = "../data/test/images"
image_size = (512, 512)

# Load the data using the function
train_images, train_labels, val_images, val_labels, test_images = load_data(
    image_train_folder, label_train_folder, image_val_folder, label_val_folder, image_test_folder, image_size
)

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, Activation, BatchNormalization, Concatenate
from tensorflow.keras.applications import VGG19
from tensorflow.keras import backend as K


def loss(y_true, y_pred):
    def dice_loss(y_true, y_pred):
        y_pred = tf.math.sigmoid(y_pred)
        numerator = 2 * tf.reduce_sum(y_true * y_pred)
        denominator = tf.reduce_sum(y_true + y_pred)
        return 1 - numerator / denominator

    y_true = tf.cast(y_true, tf.float32)
    cross_entropy_loss = tf.nn.sigmoid_cross_entropy_with_logits(y_true, y_pred)
    total_loss = cross_entropy_loss + dice_loss(y_true, y_pred)

    return tf.reduce_mean(total_loss)


def iou_metric(y_true, y_pred):
    y_pred = tf.math.round(y_pred)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    iou = intersection / (union + K.epsilon())
    return iou


class UNet:
    def __init__(self, input_shape):
        self.input_shape = input_shape
        self.model = self.build_vgg19_unet()

    def conv_block(self, input, num_filters):
        x = Conv2D(num_filters, 3, padding="same")(input)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        x = Conv2D(num_filters, 3, padding="same")(x)
        x = BatchNormalization()(x)
        x = Activation("relu")(x)

        return x

    def decoder_block(self, input, skip_features, num_filters):
        x = Conv2DTranspose(num_filters, (2, 2), strides=2, padding="same")(input)
        x = Concatenate()([x, skip_features])
        x = self.conv_block(x, num_filters)
        return x

    def build_vgg19_unet(self):
        """ Input """
        inputs = Input(self.input_shape)

        """ Pre-trained VGG19 Model """
        vgg19 = VGG19(include_top=False, weights="imagenet", input_tensor=inputs)

        """ Encoder """
        s1 = vgg19.get_layer("block1_conv2").output  ## (512 x 512)
        s2 = vgg19.get_layer("block2_conv2").output  ## (256 x 256)
        s3 = vgg19.get_layer("block3_conv4").output  ## (128 x 128)
        s4 = vgg19.get_layer("block4_conv4").output  ## (64 x 64)

        """ Bridge """
        b1 = vgg19.get_layer("block5_conv4").output  ## (32 x 32)

        """ Decoder """
        d1 = self.decoder_block(b1, s4, 512)  ## (64 x 64)
        d2 = self.decoder_block(d1, s3, 256)  ## (128 x 128)
        d3 = self.decoder_block(d2, s2, 128)  ## (256 x 256)
        d4 = self.decoder_block(d3, s1, 64)  ## (512 x 512)

        """ Output """
        outputs = Conv2D(1, 1, padding="same", activation="sigmoid")(d4)

        model = Model(inputs, outputs, name="VGG19_U-Net")
        return model

def train_and_predict_unet(input_shape, train_images, train_labels, val_images, val_labels, test_images, epochs=10, batch_size=16):
    unet = UNet(input_shape)

    # Compile the model
    unet.compile(optimizer='adam')

    # Print the model summary
    unet.model.summary()

    # Train the model
    validation_data = (val_images, val_labels)  # Assuming you have validation data
    unet.train(train_images, train_labels, epochs=epochs, batch_size=batch_size, validation_data=validation_data)

    # Predict
    test_labels_pred = unet.predict(test_images)
    return test_labels_pred


input_shape = (512, 512, 3)
test_labels_pred = train_and_predict_unet(input_shape, train_images, train_labels, val_images, val_labels, test_images, epochs=10, batch_size=16)
