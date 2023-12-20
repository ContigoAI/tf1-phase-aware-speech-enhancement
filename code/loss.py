import tensorflow as tf

# Small constant to avoid division by zero
EPS = 1e-16

def sdr_loss(X, y):
    """
    Compute Source-to-Distortion Ratio (SDR) loss between predicted and target signals.

    Args:
        X: Predicted signal
        y: Target signal

    Returns:
        SDR loss
    """
    numerator = tf.reduce_mean(X * y)
    denominator = EPS + tf.reduce_mean(tf.norm(X) * tf.norm(y))

    # Compute and return SDR loss
    return numerator / denominator

def weighted_sdr_loss(X, y_pred, y_true):
    """
    Compute weighted SDR loss considering both voice and noise components.

    Args:
        X: Original signal
        y_pred: Predicted signal
        y_true: Target signal

    Returns:
        Weighted SDR loss
    """
    # SDR loss for the voice component
    voice_target = -sdr_loss(y_true, y_pred)

    # SDR loss for the noise component
    noise_target = -sdr_loss(X - y_true, X - y_pred)

    # Weighting factor alpha based on the ratio of norms of target voice and target noise
    alpha = tf.reduce_mean(tf.norm(y_true) / (tf.norm(X - y_true) + tf.norm(y_true)))

    # Combine voice and noise losses with weights and return the result
    return alpha * voice_target + (1 - alpha) * noise_target