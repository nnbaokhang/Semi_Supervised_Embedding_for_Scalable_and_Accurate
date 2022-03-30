import tensorflow as tf
from matplotlib import pyplot as plt
from os.path import join


class ReconstructionViewer(tf.keras.callbacks.Callback):
    def __init__(self, model, out_dir):
        super(ReconstructionViewer, self).__init__()
        self.model = model
        self.out_dir = out_dir

    def on_epoch_end(self, epoch, test_data, logs):
        encoder_out = self.model.encoder(test_data[:25], training=False)
        pred = self.model.full_ae_path(encoder_out, training=False)
        # att_pred = model.constrain_ae_path(encoder_out,training=False)
        print("Prediction:", pred)
        plt.figure(figsize=(9, 7))
        plt.title("Predicted vs Actual Data (Test Set)")
        plt.plot(test_data[6].transpose().reshape(-1), label="Actual")
        plt.plot(pred[6].numpy().reshape(test_data[6].shape).transpose().reshape(-1), label="Predicted")
        # plt.plot(att_pred[6],label="Constrained")
        plt.legend()
        plt.savefig(join(self.out_dir, "reconst/rc_epoch_%d.png" % epoch))
        plt.close()
