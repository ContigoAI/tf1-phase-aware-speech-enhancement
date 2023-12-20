class Flags():
    def __init__(self):
        # Model Training
        self.LoadSavedModel = True  # Flag indicating whether to load a saved model

        # Model Parameters
        self.channels = 16384  # Number of channels
        self.threads = 1  # Number of threads
        self.epochs = 20  # Number of training epochs
        self.batch_size = 4  # Batch size
        self.validation_size = 0.05  # Fraction of data used for validation

        # Audio Processing Parameters
        self.stft_freq_samples = 512  # STFT frequency samples
        self.fs = 10e3  # Sampling frequency
        self.net_size = 2  # Size of the neural network
        self.overlap = 8  # Overlap factor
        self.noverlap = int((1 - 1.0 / self.overlap) * self.stft_freq_samples)  # Non-overlapping samples

        # File Paths
        self.check_name = "<>"  # Checkpoint name
        self.ckdir = F"<>"  # Checkpoint directory
        self.resultDir = F"<>"

        # Data Directories
        self.train_noise_dir = F"<>"  # Directory for training noise data
        self.train_clean_dir = F"<>"  # Directory for training clean data

        # Model Training Hyperparameters
        self.regulizer_weight = 1e-1  # Regularization weight
        self.learning_rate = 5e-3  # Initial learning rate
        self.end_learning_rate = 1e-5  # Final learning rate
