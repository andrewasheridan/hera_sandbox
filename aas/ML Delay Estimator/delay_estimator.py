from network_design import faora
from network_data_scaling import *

import tensorflow as tf

def estimate_delay(samples, network, model_path = 'trained_network_model/trained_model.ckpt-2549'):
    """Estimate the delay in a sample. Min delay is -0.040, Max delay is 0.040.

    Parameters
    ----------
    samples: Each element in samples is a 2D array of angle data with shape (60,1024)

     """

    saver = tf.train.Saver()
    with tf.Session() as session:

        # load in saved model
        # TODO: Suppress load statement..
        saver.restore(session,model_path)

        predictions = []
        for sample in samples:
            inputs = angle_tx(sample)
            scaled_prediction = session.run(network.prediction, feed_dict = {network.X: inputs, network.keep_prob : 1.00})
            unscaled_prediction = delay_itx(scaled_prediction.T[0])
            predictions.append(unscaled_prediction)

    return predictions