from .config import Flags
from .utils import audio_generator_complex, config_dataset, get_graph_size, get_wav_files
from .loss import weighted_sdr_loss
from .network import make_asppunet_3D
import tensorflow as tf
import datetime
import sklearn
import os
import time
import numpy as np
import scipy
import random

def make_train_op(X, y_pred, y_true, flags, additional_loss_input):
    """
    Create the training operation.

    Args:
        X: Input tensor.
        y_pred: Predicted output tensor.
        y_true: True output tensor.
        flags: Flags object.
        additional_loss_input: Additional loss input tensor.

    Returns:
        Tuple containing the training operation and the loss tensor.
    """
    # Loss Calculation:
    loss = weighted_sdr_loss(X, y_pred, y_true)
    tf.summary.scalar("weighted_sdr_loss", loss)

    # MSE Loss
    if additional_loss_input is not None:
        frame_step = flags.stft_freq_samples - flags.noverlap - 2
        stft_true = tf.contrib.signal.stft(y_true, frame_length=flags.stft_freq_samples, frame_step=frame_step)
        mag_true = tf.abs(stft_true)
        mag_loss = tf.reduce_mean(tf.abs(mag_true - additional_loss_input))
        loss += mag_loss
        tf.summary.scalar("mag_loss", mag_loss)

    # Global Step and Learning Rate Decay
    global_step = tf.train.get_or_create_global_step()
    tf.summary.scalar("global_step", global_step)

    starter_learning_rate = flags.learning_rate
    end_learning_rate = flags.end_learning_rate
    learning_rate = tf.train.polynomial_decay(starter_learning_rate, global_step,
                                              flags.fs, end_learning_rate,
                                              power=0.5)
    tf.summary.scalar("learning_rate", learning_rate)

    # Optimizer and Minimization
    optim = tf.train.AdamOptimizer(learning_rate=learning_rate)
    return optim.minimize(loss, global_step=global_step), loss

def get_train_data(flags):
    """
    Get training and validation data.

    Args:
        flags: Flags object.

    Returns:
        Tuple containing iterator, training dataset, validation dataset, training size, and validation size.
    """
    all_files = get_wav_files(flags.train_clean_dir)
    train_files, valid_files = sklearn.model_selection.train_test_split(all_files, test_size=flags.validation_size, random_state=42)
    train = lambda: audio_generator_complex(train_files, flags)
    valid = lambda: audio_generator_complex(valid_files, flags)

    with tf.name_scope('input'):
        input_shape = tuple(np.array([None]))
        output_shape = input_shape
        train_images_tf = tf.data.Dataset.from_generator(train, (tf.float32, tf.float32), (input_shape, output_shape))
        valid_images_tf = tf.data.Dataset.from_generator(valid, (tf.float32, tf.float32), (input_shape, output_shape))
        train_images_tf = config_dataset(train_images_tf, flags)
        valid_images_tf = config_dataset(valid_images_tf, flags)
        iterator = tf.data.Iterator.from_structure(train_images_tf.output_types, train_images_tf.output_shapes)

    # Get datasets sizes
    train_size = sum(samples_per_file for _, _, samples_per_file in train())
    valid_size = sum(samples_per_file for _, _, samples_per_file in valid())
    return iterator, train_images_tf, valid_images_tf, train_size, valid_size


def save_loss(epoch_array, validation_accuracy, train_accuracy, path, train_loss_arr):
    """
    Save loss information to files.

    Args:
        epoch_array: Array containing epoch numbers.
        validation_accuracy: Validation accuracy values.
        train_accuracy: Training accuracy values.
        path: Path to save the files.
        train_loss_arr: Training loss values.
    """
    comb_ = np.asarray([epoch_array, validation_accuracy, train_accuracy])
    np.savetxt(os.path.join(path, "loss.csv"), comb_, delimiter=",")
    np.savetxt(os.path.join(path, "train_loss.csv"), train_loss_arr, delimiter=",")


def main():
    # Initialize Flags object
    flags = Flags()
    # Clears the default graph stack and resets the global default graph
    tf.reset_default_graph()
    graph = tf.get_default_graph()

    # Get training and validation data
    iterator, train_images_tf, valid_images_tf, train_size, valid_size = get_train_data(flags)
    n_batches_train = int(train_size // flags.batch_size)
    n_batches_valid = int(valid_size // flags.batch_size)

    # Define input placeholders and build the UNET model
    X, y = iterator.get_next()
    mode = tf.placeholder(tf.bool, name="mode")
    pred, _, _ = make_asppunet_3D(X, mode, flags, features=flags.net_size, last_pad=True, mask=True)
    additional_loss_input = None                                                
    print("Defined UNET")

    # Build the training operation
    with tf.name_scope('optimize'):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op, loss = make_train_op(X, pred, y, flags, additional_loss_input)

    # Merge all summaries
    summary_op = tf.summary.merge_all()

    # Define checkpoint directory
    checkpoint_dir = os.path.join(flags.ckdir, str(get_graph_size())[:3] + '_' + str(time.time()))        

    # Initialize TensorFlow session

    # gpu_options = tf.GPUOptions(allow_growth=True)
    # config = tf.ConfigProto(log_device_placement=False, gpu_options=gpu_options,
    #                         inter_op_parallelism_threads=flags.threads,intra_op_parallelism_threads=flags.threads)
    
    with tf.Session(graph=graph) as sess: # with tf.Session(graph=graph, config=config) as sess:
        start_time = time.time()

        sess.run(tf.local_variables_initializer())
        sess.run(tf.global_variables_initializer())

        # Create a Saver for saving and restoring variables
        saver = tf.train.Saver(tf.global_variables(), max_to_keep=1)

        # Check for the latest checkpoint
        check_name = flags.check_name
        latest_check_point = tf.train.latest_checkpoint(os.path.join(flags.ckdir, check_name))

        if latest_check_point is not None and flags.LoadSavedModel:
            saver.restore(sess, latest_check_point)
            print('Restore model, checkpoint:' + latest_check_point)
        else:
            print('Define checkpoint dir:' + checkpoint_dir)

        # Get the global step
        global_step = tf.train.get_global_step(sess.graph)
        best_val_loss = 1e10

        # Print training information
        print("Training {} examples validating on {} examples".format(train_size, valid_size))
        loss_valid_arr, loss_train_arr, epoch_arr = [], [], []
        loss_arr = [] 

        # Iterate through epochs
        for epoch in range(flags.epochs):
            epoch_arr.append(epoch + 1)
            print("Epoch {}/ {}". format(epoch+1,flags.epochs))

            # Shuffle and initialize training dataset
            ds = train_images_tf.shuffle(128, seed=random.randint(0, 1024))
            sess.run(iterator.make_initializer(ds))
            sess.run(tf.local_variables_initializer())

            # Iterate through batches
            mean_loss = 0
            for i in range(n_batches_train):
                try:
                    _, l, step_summary, global_step_value, predicted = sess.run(
                        [train_op, loss, summary_op, global_step, pred],
                        feed_dict={mode: True})
                    mean_loss += l
                    loss_arr.append(l)

                except Exception as e:
                    msg = 'Finished epoch with expectation {}'.format(e)
                    print(msg)

            mean_loss /= n_batches_train
            loss_train_arr.append(mean_loss)

            # Initialize validation dataset
            sess.run(iterator.make_initializer(valid_images_tf))
            sess.run(tf.local_variables_initializer())

            # Iterate through validation batches
            mean_valid_loss = 0
            for i in range(n_batches_valid):
                try:
                    predicted, l, step_summary = sess.run([pred, loss, summary_op],feed_dict={mode: True})
                    mean_valid_loss += l

                except Exception as e:
                    msg = "finished epoch with exception {}".format(e)
                    print(msg)

            mean_valid_loss /= n_batches_valid
            loss_valid_arr.append(mean_valid_loss)
            msg = "Finished epoch :{} at {}. {}".format(epoch,datetime.datetime.now(),mean_valid_loss)
            print(msg)

            # Save model if validation loss improves
            if mean_valid_loss < best_val_loss:
                saver.save(sess, "{}/model_{}.ckpt".format(checkpoint_dir, mean_valid_loss))
                print("SAVE : {}/model_{}.ckpt".format(checkpoint_dir, mean_valid_loss))
                
            save_loss(epoch_arr, loss_valid_arr, loss_train_arr, checkpoint_dir, loss_arr)

if __name__=='__main__':
    main()
    print("Done")