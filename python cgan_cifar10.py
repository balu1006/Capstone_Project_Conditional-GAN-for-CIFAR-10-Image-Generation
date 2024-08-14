import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Reshape, Flatten, Dropout, Input, Embedding, multiply, Concatenate
from tensorflow.keras.layers import BatchNormalization, Activation, LeakyReLU, UpSampling2D, Conv2D
from tensorflow.keras.optimizers import Adam
import numpy as np
import matplotlib.pyplot as plt

# Load CIFAR-10 data
(x_train, y_train), (_, _) = cifar10.load_data()

# Normalize the images to the range [-1, 1]
x_train = (x_train.astype(np.float32) - 127.5) / 127.5

# Generator model
def build_generator():
    noise_input = Input(shape=(100,))
    label_input = Input(shape=(1,), dtype='int32')

    # Embedding layer for the label
    label_embedding = Flatten()(Embedding(10, 100)(label_input))

    # Concatenate noise and label embeddings
    model_input = Concatenate()([noise_input, label_embedding])

    model = Sequential()
    model.add(Dense(256, input_dim=200))  # Corrected input dimension
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(512))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(1024))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(2048))
    model.add(LeakyReLU(alpha=0.2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Dense(32*32*3, activation='tanh'))
    model.add(Reshape((32, 32, 3)))

    img = model(model_input)

    return Model([noise_input, label_input], img)

# Discriminator model
def build_discriminator():
    img_input = Input(shape=(32, 32, 3))
    label_input = Input(shape=(1,), dtype='int32')

    # Embedding layer for the label
    label_embedding = Flatten()(Embedding(10, np.prod((32, 32, 3)))(label_input))

    # Flatten the image
    flat_img = Flatten()(img_input)

    # Concatenate flattened image and label embeddings
    model_input = multiply([flat_img, label_embedding])

    model = Sequential()
    model.add(Dense(512, input_dim=np.prod((32, 32, 3))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(256))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.4))
    model.add(Dense(1, activation='sigmoid'))

    validity = model(model_input)
    
    return Model([img_input, label_input], validity)

# Compile models
def compile_models(generator, discriminator):
    optimizer = Adam(0.0002, 0.5)
    discriminator.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])

    noise = Input(shape=(100,))
    label = Input(shape=(1,))
    img = generator([noise, label])
    discriminator.trainable = False
    validity = discriminator([img, label])

    combined = Model([noise, label], validity)
    combined.compile(loss='binary_crossentropy', optimizer=optimizer)
    return combined

# Training function
def train(generator, discriminator, combined, epochs, batch_size, save_interval):
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    for epoch in range(epochs):

        # Train Discriminator
        idx = np.random.randint(0, x_train.shape[0], batch_size)
        imgs, labels = x_train[idx], y_train[idx]

        noise = np.random.normal(0, 1, (batch_size, 100))
        gen_imgs = generator.predict([noise, labels])

        d_loss_real = discriminator.train_on_batch([imgs, labels], valid)
        d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # Train Generator
        sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
        g_loss = combined.train_on_batch([noise, sampled_labels], valid)

        print(f"{epoch} [D loss: {d_loss[0]} | D accuracy: {100*d_loss[1]}] [G loss: {g_loss}]")

        if epoch % save_interval == 0:
            save_imgs(generator, epoch)

# Save generated images
def save_imgs(generator, epoch, examples=10):
    noise = np.random.normal(0, 1, (examples, 100))
    sampled_labels = np.arange(0, examples).reshape(-1, 1)

    gen_imgs = generator.predict([noise, sampled_labels])

    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(1, examples, figsize=(examples, 1))
    for i in range(examples):
        axs[i].imshow(gen_imgs[i])
        axs[i].axis('off')
    plt.show()

# Generate images
def generate_images(generator, class_label):
    noise = np.random.normal(0, 1, (1, 100))
    label = np.array([class_label])
    gen_img = generator.predict([noise, label])

    gen_img = 0.5 * gen_img + 0.5
    plt.imshow(gen_img[0])
    plt.show()

# Run the workflow
generator = build_generator()
discriminator = build_discriminator()
combined = compile_models(generator, discriminator)
train(generator, discriminator, combined, epochs=10000, batch_size=64, save_interval=100)
class_label = int(input("Enter the class label (0-9): "))
generate_images(generator, class_label)
