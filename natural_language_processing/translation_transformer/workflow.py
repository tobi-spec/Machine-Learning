import tensorflow_datasets as tfds
import tensorflow as tf
import tensorflow_text
from utils import *
from yaml_parser import get_hyperparameters
from model import Transformer

setup_gpu_usage()
hyperparameters: dict = get_hyperparameters("hyperparameters.yaml")

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en',
                               with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

model_name = 'ted_hrlr_translate_pt_en_converter'
tf.keras.utils.get_file(
    f'{model_name}.zip',
    f'https://storage.googleapis.com/download.tensorflow.org/models/{model_name}.zip',
    cache_dir='.', cache_subdir='', extract=True
)
tokenizers = tf.saved_model.load(model_name)


def prepare_batch(pt, en):
    pt = tokenizers.pt.tokenize(pt)  # Output is ragged.
    pt = pt[:, :hyperparameters["max_tokens"]]  # Trim to MAX_TOKENS.
    pt = pt.to_tensor()  # Convert to 0-padded dense Tensor

    en = tokenizers.en.tokenize(en)
    en = en[:, :(hyperparameters["max_tokens"] + 1)]
    en_inputs = en[:, :-1].to_tensor()  # Drop the [END] tokens
    en_labels = en[:, 1:].to_tensor()  # Drop the [START] tokens

    return (pt, en_inputs), en_labels


def make_batches(dataset):
    return (
        dataset
            .shuffle(hyperparameters["buffer_size"])
            .batch(hyperparameters["batch_size"])
            .map(prepare_batch, tf.data.AUTOTUNE)
            .prefetch(buffer_size=tf.data.AUTOTUNE)
    )


train_batches = make_batches(train_examples)
val_batches = make_batches(val_examples)

transformer = Transformer(
    num_layers=hyperparameters["number_of_layers"],
    d_model=hyperparameters["d_model"],
    num_heads=hyperparameters["number_of_heads"],
    dff=hyperparameters["dff"],
    input_vocab_size=tokenizers.pt.get_vocab_size().numpy(),
    target_vocab_size=tokenizers.en.get_vocab_size().numpy(),
    dropout_rate=hyperparameters["dropout_rate"])


optimizer = tf.keras.optimizers.Adam(learning_rate=CustomSchedule(hyperparameters["d_model"]),
                                     beta_1=0.9,
                                     beta_2=0.98,
                                     epsilon=1e-9)

transformer.compile(
    loss=masked_loss,
    optimizer=optimizer,
    metrics=[masked_accuracy])

transformer.fit(train_batches,
                epochs=20,
                validation_data=val_batches)

transformer.save("./", save_format='tf')
