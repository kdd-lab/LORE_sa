import base64
import io
import os
import unittest

import keras
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import layers, Model
from tensorflow.keras import regularizers


@tf.keras.utils.register_keras_serializable()
class VanillaVAE(Model):
    def __init__(self, in_channels=1, latent_dim=128, hidden_dims=None, **kwargs):
        super(VanillaVAE, self).__init__(**kwargs)

        self.latent_dim = latent_dim
        self.dropout_rate = 0.2  # You can experiment with different dropout rates
        self.l2_reg = 1e-5
        self.noise_layer = layers.GaussianNoise(0.3)
        self.in_channels = in_channels

        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]

        # Build Encoder
        encoder_layers = []
        for h_dim in hidden_dims:
            encoder_layers.append(layers.Conv2D(h_dim, kernel_size=3, strides=2, padding='same',
                                                kernel_regularizer=regularizers.l2(self.l2_reg)))  # L2 regularization
            encoder_layers.append(layers.BatchNormalization())
            encoder_layers.append(layers.LeakyReLU())
            encoder_layers.append(layers.Dropout(self.dropout_rate))  # Dropout added

        self.encoder = tf.keras.Sequential(encoder_layers)

        self.fc_mu = layers.Dense(latent_dim, kernel_regularizer=regularizers.l2(self.l2_reg))
        self.fc_var = layers.Dense(latent_dim, kernel_regularizer=regularizers.l2(self.l2_reg))

        # Build Decoder
        decoder_layers = []
        self.decoder_input = layers.Dense(2 * 2 * 512, kernel_regularizer=regularizers.l2(self.l2_reg))  # Restoring the Dense layer size

        hidden_dims.reverse()

        for i in range(len(hidden_dims) - 1):
            decoder_layers.append(
                layers.Conv2DTranspose(hidden_dims[i + 1], kernel_size=3, strides=2, padding='same', output_padding=1))
            decoder_layers.append(layers.BatchNormalization())
            decoder_layers.append(layers.LeakyReLU())

        self.decoder = tf.keras.Sequential(decoder_layers)

        self.final_layer = tf.keras.Sequential([
            layers.Conv2DTranspose(hidden_dims[-1], kernel_size=3, strides=2, padding='same'),
            layers.BatchNormalization(),
            layers.LeakyReLU(),
            layers.Conv2DTranspose(in_channels, kernel_size=3, padding='same'),
            layers.Activation('sigmoid')
        ])

    def encode(self, x):
        x = self.noise_layer(x)  # Apply noise to the input
        x = self.encoder(x)
        x = layers.Flatten()(x)
        mu = self.fc_mu(x)
        log_var = self.fc_var(x)
        return mu, log_var

    def reparameterize(self, mu, log_var):
        eps = tf.random.normal(shape=tf.shape(mu))
        return eps * tf.exp(0.5 * log_var) + mu

    def decode(self, z):
        x = self.decoder_input(z)
        x = layers.Reshape((2, 2, 512))(x)
        x = self.decoder(x)
        x = self.final_layer(x)

        # Ensure the shape matches the input
        x = tf.reshape(x, [-1, 64, 64, self.in_channels])

        return x

    def call(self, inputs):
        mu, log_var = self.encode(inputs)
        z = self.reparameterize(mu, log_var)
        return self.decode(z), mu, log_var

    def get_config(self):
        config = super(VanillaVAE, self).get_config()
        config.update({
            'in_channels': self.in_channels,
            'latent_dim': self.latent_dim,
            'hidden_dims': [32, 64, 128, 256, 512] # [32, 64, 128, 256, 512]
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)

# Custom loss function
@tf.keras.utils.register_keras_serializable()
def vae_loss(inputs, outputs, mu, log_var, kld_weight=0.0001):
    recons_loss = tf.reduce_mean(tf.square(inputs - outputs))
    log_var = tf.clip_by_value(log_var, -10, 10)
    kld_loss = -0.5 * tf.reduce_mean(1 + log_var - tf.square(mu) - tf.exp(log_var), axis=-1)
    total_loss = recons_loss + kld_weight * tf.reduce_mean(kld_loss)
    return total_loss

@tf.keras.utils.register_keras_serializable()
class CustomVAE(keras.Model):
    def __init__(self, vae, **kwargs):
        super(CustomVAE, self).__init__(**kwargs)
        self.vae = vae
        self.encoder = vae.encoder  # Expose the encoder
        self.decoder = vae.decoder  # Expose the decoder

    def train_step(self, data):
        if isinstance(data, tuple):
            inputs = data[0]
        else:
            inputs = data

        with tf.GradientTape() as tape:
            outputs, mu, log_var = self.vae(inputs)
            loss = vae_loss(inputs, outputs, mu, log_var)

        grads = tape.gradient(loss, self.vae.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.vae.trainable_weights))

        return {"loss": loss}

    def call(self, inputs):
        return self.vae(inputs)

    def get_config(self):
        config = super(CustomVAE, self).get_config()
        config.update({
            "vae": self.vae.get_config(),  # Save the config of the vae (VanillaVAE)
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Recreate the vae (VanillaVAE) from its config
        vae_config = config.pop("vae")
        vae = VanillaVAE.from_config(vae_config)  # Recreate VanillaVAE from its config
        return cls(vae, **config)

# Custom loss function for Keras compile
@tf.keras.utils.register_keras_serializable()
def custom_vae_loss(inputs, outputs):
    return vae_loss(inputs, outputs[0], outputs[1], outputs[2])


def load_keras_encoder(dataset="pneumonia", latent_dim=128, pxl=128, in_channels=3):
    """
        Load the Keras model and the training dataset

    :param latent_dim: the dimension of the latent space of the model to be used

    :return: the autoencoder model and the training dataset
    """
    my_model_path = f"../../MNISTFastAPI/model/vae/vae_{dataset}_{pxl}_{latent_dim}.keras"

    if not os.path.exists(my_model_path):
        raise OSError(f"SavedModel file does not exist at: {my_model_path}")

    vae_model = tf.keras.models.load_model(
        my_model_path,
        custom_objects={
            "VanillaVAE": VanillaVAE,
            "vae_loss": vae_loss,
            "CustomVAE": CustomVAE
        }
    )

    # print(vae_model.summary())
    return vae_model

def load_keras_model(dataset="pneumonia", pxl=128):
    my_model_path=f"../../MNISTFastAPI/model/bb/{dataset}_{pxl}.keras"
    if not os.path.exists(my_model_path):
        raise OSError(f"SavedModel file does not exist at: {my_model_path}")
    bb_model = tf.keras.models.load_model(my_model_path)
    return bb_model


def img2base64(img):
    """Convert a TensorFlow tensor image (batch, H, W, C) to a base64-encoded PNG string."""

    # Convert TensorFlow tensor to NumPy array if needed
    if isinstance(img, tf.Tensor):
        img = img.numpy()

        # Remove batch dimension if present (e.g., (1, 64, 64, 3) → (64, 64, 3))
    if img.shape[0] == 1:
        img = np.squeeze(img, axis=0)

    # Ensure image data is in uint8 format
    if img.dtype != np.uint8:
        img = (img * 255).astype(np.uint8)  # Scale values if they're in range [0,1]

    # Convert NumPy array to PIL image
    pil_image = Image.fromarray(img)

    # Save the image to a buffer
    buffer = io.BytesIO()
    pil_image.save(buffer, format='PNG', optimize=True)

    # Encode as base64
    encoded_string = base64.b64encode(buffer.getvalue()).decode('utf-8')

    return encoded_string

class LOREImageTest(unittest.TestCase):
    def setUp(self):
        self.latent_dim = 128
        self.pxl = 64
        self.model_name = "blood"
        self.channels = 3
        # print available GPU devices
        print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
    
    def test_loading_model_vae(self):
        blood_64_128 = load_keras_encoder(self.model_name, latent_dim=self.latent_dim, pxl=self.pxl)
        self.assertIsNotNone(blood_64_128)

    def test_decode_random_point_to_image(self):
        blood_64_128 = load_keras_encoder(self.model_name, latent_dim=self.latent_dim, pxl=self.pxl)
        random_point = np.random.normal(size=(1, self.latent_dim))
        img = blood_64_128.vae.decode(random_point)
        self.assertEqual(img.shape, (1, self.pxl, self.pxl, self.channels))


    def test_decode_random_point_and_classify(self):
        blood_64_128 = load_keras_encoder(self.model_name, latent_dim=self.latent_dim, pxl=self.pxl)
        random_point = np.random.normal(size=(1, self.latent_dim))
        img = blood_64_128.vae.decode(random_point)
        bb_model=load_keras_model(self.model_name, self.pxl)
        prediction_proba = bb_model.predict(img)
        # compute argmax of the prediction
        prediction = np.argmax(prediction_proba)

        print(prediction)
        enc_img = img2base64(img)
        print(enc_img)

        assert prediction is not None


if __name__ == '__main__':
    unittest.main()
