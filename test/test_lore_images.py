import base64
import unittest
from io import BytesIO

import mlflow
import keras
import mlflow.keras
import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras import Model, layers, regularizers

from lore_sa.bbox.keras_ts_classifier_wrapper import keras_ts_classifier_wrapper
from lore_sa.encoder_decoder import EncDec
from lore_sa.neighgen import GeneticGenerator
from lore_sa.surrogate import DecisionTreeSurrogate
from lore_sa.lore import Lore


@tf.keras.utils.register_keras_serializable()
class VanillaVAE(Model):
    def __init__(self, in_channels=3, latent_dim=128, hidden_dims=None, dropout_rate=0.2, l2_reg=1e-5, **kwargs):
        super(VanillaVAE, self).__init__(**kwargs)

        self.latent_dim = latent_dim
        self.in_channels = in_channels
        self.dropout_rate = dropout_rate
        self.l2_reg = l2_reg
        self.noise_layer = layers.GaussianNoise(0.3)

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
            layers.Conv2DTranspose(self.in_channels, kernel_size=3, padding='same'),
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
            'hidden_dims': [32, 64, 128, 256, 512],  # [32, 64, 128, 256, 512]
            'dropout_rate': self.dropout_rate,
            'l2_reg': self.l2_reg
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

def image_to_base64(image_array):
    """
    Convert a numpy image array to base64 string.

    Args:
        image_array: Numpy array with shape (H, W, C) and values in [0, 255]

    Returns:
        Base64 encoded string of the PNG image
    """
    # Ensure the image is in the correct format (uint8)
    if image_array.max() <= 1.0:
        image_array = (image_array * 255).astype(np.uint8)
    else:
        image_array = image_array.astype(np.uint8)

    # Normalize shape for PIL (grayscale can be HxW; RGB is HxWx3)
    if image_array.ndim == 3 and image_array.shape[-1] == 1:
        image_array = np.squeeze(image_array, axis=-1)
    elif image_array.ndim not in (2, 3):
        raise ValueError(f"Unsupported image shape for base64 conversion: {image_array.shape}")

    # Convert to PIL Image
    pil_image = Image.fromarray(image_array)

    # Save to BytesIO buffer as PNG
    buffer = BytesIO()
    pil_image.save(buffer, format='PNG')
    buffer.seek(0)

    # Encode to base64
    base64_str = base64.b64encode(buffer.read()).decode('utf-8')

    return base64_str

class CustomVAEEncoderDecoder(EncDec):
    def __init__(self, vae_model):
        super().__init__(dataset_descriptor=None)  # No dataset descriptor needed for this encoder/decoder
        self.vae = vae_model
        self.type = 'vae'  # Set type to 'vae' to indicate this is a VAE-based encoder/decoder

    def encode(self, x: np.array):
        # Encode the input using the VAE's encoder
        # Ensure input is a numpy array and convert to tensor
        x = np.array(x)
        x = tf.convert_to_tensor(x, dtype=tf.float32)
        mu, log_var = self.vae.vae.encode(x)
        return mu.numpy()

    def decode(self, z: np.array):
        # Decode the latent representation using the VAE's decoder
        decoded_images = self.vae.vae.decode(z)
        return decoded_images.numpy()  # Return the decoded images as numpy arrays

    def encode_target_class(self, x: np.array):
        # This method is not needed for image data, but we can return the input as is
        return x #Return the input as is for encoding target class

    def decode_target_class(self, z: np.array):
        # This method is not needed for image data, but we can return the input as is
        return z #Return the input as is for decoding target class


class LoreImagesTest(unittest.TestCase):

    def setUp(self):
        dataset = 'mnist'
        pxl_size = 64
        latent_dim = 4
        mlflow_uri = 'http://localhost:5000'

        mlflow.set_tracking_uri(mlflow_uri)

        # 1. prepare dataset
        from tensorflow.keras.datasets import mnist
        (_, _), (x_test, y_test) = mnist.load_data()
        x_test = tf.image.resize(tf.expand_dims(x_test, -1), (pxl_size, pxl_size)).numpy() / 255.0
        self.images = x_test

        # 2. prepare encoders
        vae_model_name = f"{dataset}_vae_p{pxl_size}_l{latent_dim}"
        vae_model_uri = f"models:/{vae_model_name}@champion"

        self.enc = CustomVAEEncoderDecoder(mlflow.keras.load_model(vae_model_uri))

        # 3. load model
        model_name = f"{dataset}_classifier_{pxl_size}"
        model_uri = f"models:/{model_name}@champion"
        self.model = mlflow.keras.load_model(model_uri)


        # # 4. wrapper for the model
        self.bbox = keras_ts_classifier_wrapper(self.model)
        #
        # # 5. prepare final lore
        # encoder = ColumnTransformerEnc(dataset.descriptor)
        generator = GeneticGenerator(self.bbox, None, self.enc, 0.1)
        surrogate = DecisionTreeSurrogate()
        #
        self.imageLore= Lore(self.bbox, None, self.enc, generator, surrogate)




    def test_lore_images_init(self):
        # given
        # assert self.images.shape == (10000, 64, 64, 1)
        # assert self.enc is not None
        # assert self.model is not None
        # # when
        # num_samples = 10
        # rnd_idx = np.random.randint(0, self.images.shape[0], num_samples)
        # sample_images = self.images[rnd_idx]
        #
        # for i in range(num_samples):
        #     base64_str = image_to_base64(sample_images[i])
        #     assert isinstance(base64_str, str)
        #     assert len(base64_str) > 0
        #     print(f"{base64_str}")
        #
        # predictions = self.bbox.predict_proba(sample_images) # Predict on the first 10 images
        # predicted_class = np.argmax(predictions, axis=1)
        # confidence = np.max(predictions, axis=1)
        # less_confident_index = np.argmin(confidence)
        # print(f"Least confident prediction index: {less_confident_index}, Confidence: {confidence[less_confident_index]}")
        # print("Predicted classes:", predicted_class)
        # print("Confidence scores:", confidence)
        # assert predictions.shape == (num_samples, 10) # Assuming 10 classes for MNIST
        # assert predicted_class.shape == (num_samples,)
        # assert confidence.shape == (num_samples,)
        #
        # mu = self.enc.encode(sample_images)
        # print(f"Encoded images shape: {mu}")
        # assert mu.shape == (num_samples, 4) # Check if the encoded shape matches the latent dimension of the VAE
        #
        # reconstructed_images = self.enc.decode(mu)
        # print(f"Reconstructed images shape: {reconstructed_images.shape}")
        # assert reconstructed_images.shape == (num_samples, 64, 64, 1)

        explanation = self.imageLore.explain(self.images[0], num_instances=100)
        print(explanation)