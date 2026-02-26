"""
Ben Hayes 2020

ECS7006P Music Informatics

Coursework 1: Beat Tracking

File: beat_tracking_tcn/models/beat_net.py
Description: A CNN including a Temporal Convolutional Layer designed to predict
             a vector of beat activations from an input spectrogram (tested
             with mel spectrograms).
"""
import torch
import torch.nn as nn
import sys
# sys.path.append('/home/nikhil/projects/dbtracker/')
from torch.nn.utils.parametrizations import weight_norm


class NonCausalTemporalLayer(nn.Module):
    """
    Implements a non-causal temporal block. Based off the model described in
    Bai et al 2018, but with the notable difference that the dilated temporal
    convolution is non-causal.

    Also implements a parallel residual convolution, allowing detail from the

    original signal to influence the output.
    """
    def __init__(
            self,
            inputs,
            outputs,
            dilation,
            kernel_size=5,
            stride=1,
            padding=4,
            dropout=0.1):
        """
        Construct an instance of NonCausalTemporalLayer

        Arguments:
            inputs {int} -- Input size in samples
            outputs {int} -- Output size in samples
            dilation {int} -- Size of dilation in samples

        Keyword Arguments:
            kernel_size {int} -- Size of convolution kernel (default: {5})
            stride {int} -- Size of convolution stride (default: {1})
            padding {int} -- How much padding to apply in total. Note, this is
                             halved and applied equally at each end of the
                             convolution to make the model non-causal.
                             (default: {4})
            dropout {float} -- The probability of dropping out a connection
                               during training. (default: {0.1})
        """
        super(NonCausalTemporalLayer, self).__init__()

        self.conv1 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv1 = weight_norm(self.conv1)
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
                inputs,
                outputs,
                kernel_size,
                stride=stride,
                padding=int(padding / 2),
                dilation=dilation)
        self.conv2 = weight_norm(self.conv2)
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)

        # self.conv3 = nn.Conv1d(
        #         inputs,
        #         outputs,
        #         kernel_size,
        #         stride=stride,
        #         padding=int(padding / 2),
        #         dilation=dilation)
        # self.elux = nn.ELU()
        # self.dropoutx = nn.Dropout(dropout)


        self.downsample = nn.Conv1d(inputs, outputs, 1)\
            if inputs != outputs else None
        self.elu3 = nn.ELU()

        self._initialise_weights(self.conv1, self.conv2, self.downsample)

    def forward(self, x):
        """
        Feed a tensor forward through the layer.

        Arguments:
            x {torch.Tensor} -- A PyTorch tensor of size specified in the
                                constructor.

        Returns:
            torch.Tensor -- A PyTorch tensor of size specified in the
                            constructor.
        """
        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)

        # y = self.conv3(y)
        # y = self.elux(y)
        # y = self.dropoutx(y)
        

        if self.downsample is not None:
            y = y + self.downsample(x)

        y = self.elu3(y)

        return y

    def _initialise_weights(self, *layers):
        for layer in layers:
            if layer is not None:
                layer.weight.data.normal_(0, 0.01)


class NonCausalTemporalConvolutionalNetwork(nn.Module):
    """
    Implements a non-causal temporal convolutional network. Based off the model
    described in Bai et al 2018. Initialises and forwards through a number of
    NonCausalTemporalLayer instances, depending on construction parameters.
    """
    def __init__(self, inputs, channels, kernel_size=5, dropout=0.1):
        """
        Construct a NonCausalTemporalConvolutionalNetwork.

        Arguments:
            inputs {int} -- Network input length.
            channels {list[int]} -- List containing number of channels each
                                    constituent temporal layer should have.

        Keyword Arguments:
            kernel_size {int} -- Size of dilated convolution kernels.
                                 (default: {5})
            dropout {float} -- The probability of dropping out a connection
                               during training. (default: {0.1})
        """
        super(NonCausalTemporalConvolutionalNetwork, self).__init__()

        self.layers = []
        n_levels = len(channels)

        for i in range(n_levels):
            dilation = 2 ** i

            n_channels_in = channels[i - 1] if i > 0 else inputs
            n_channels_out = channels[i]

            self.layers.append(
                NonCausalTemporalLayer(
                    n_channels_in,
                    n_channels_out,
                    dilation,
                    kernel_size,
                    stride=1,
                    padding=(kernel_size - 1) * dilation,
                    dropout=dropout
                )
            )
        
        self.net = nn.Sequential(*self.layers)

    def forward(self, x):
        """
        Feed a tensor forward through the network.

        Arguments:
            x {torch.Tensor} -- A PyTorch tensor of size specified in the
                                constructor.

        Returns:
            torch.Tensor -- A PyTorch tensor of size determined by the final
                            temporal convolutional layer.
        """
        y = self.net(x)
        return y
    
# from beat_tracking_tcn.models.frontend import BeatThis
# from beat_tracking_tcn.models.ctcn import TemporalConvNet

class BeatNet_offlinetcn(nn.Module):
    """ 
    PyTorch implementation of a BeatNet CNN. The network takes a
    mel-spectrogram input. It then learns an intermediate convolutional
    representation, and finally applies a non-causal Temporal Convolutional
    Network to predict a beat activation vector.

    The structure of this network is based on the model proposed in Davies &
    Bock 2019.
    """

    def __init__(
            self,
            input=(3000, 314),
            output=3000,
            channels=16,
            tcn_kernel_size=5,
            dropout=0.1,
            downbeats=False):
        """
        Construct an instance of BeatNet.

        Keyword Arguments:
            input {tuple} -- Input dimensions (default: {(3000, 81)})
            output {int} -- Output dimensions (default: {3000})
            channels {int} -- Convolution channels (default: {16})
            tcn_kernel_size {int} -- Size of dilated convolution kernels.
                                     (default: {5})
            dropout {float} -- Network connection dropout probability.
                               (default: {0.1})
        """
        super(BeatNet_offlinetcn, self).__init__()

        self.conv1 = nn.Conv2d(1, channels, (3, 3), padding=(1, 0))
        self.elu1 = nn.ELU()
        self.dropout1 = nn.Dropout(dropout)
        self.pool1 = nn.MaxPool2d((1, 5))

        self.conv2 = nn.Conv2d(channels, channels, (3, 3), padding=(1, 0))
        self.elu2 = nn.ELU()
        self.dropout2 = nn.Dropout(dropout)
        self.pool2 = nn.MaxPool2d((1, 5))

        self.conv3 = nn.Conv2d(channels, channels, (1, 8))
        self.elu3 = nn.ELU()
        self.dropout3 = nn.Dropout(dropout)
        self.pool3 = nn.MaxPool2d((1, 5))

        self.tcn = NonCausalTemporalConvolutionalNetwork(
            channels,
            [channels] * 11,
            tcn_kernel_size,
            dropout)
        

        
        # self.beatthis = BeatThis(
        #     transformer_dim=16,
        #     ff_mult=4,
        #     n_layers= 6,
        #     head_dim=4,
        #     dropout={"transformer": 0.2} 

        # )

        self.out = nn.Conv1d(16, 1 if not downbeats else 2, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, return_embeddings = True):
        """
        Feed a tensor forward through the BeatNet.

        Arguments:
            x {torch.Tensor} -- A PyTorch tensor of size specified in the
                                constructor.

        Returns:
            torch.Tensor -- A PyTorch tensor of size specified in the
                            constructor.
        """
        # print(x.shape)
        y = self.conv1(x)
        y = self.elu1(y)
        y = self.dropout1(y)
        y = self.pool1(y)

        y = self.conv2(y)
        y = self.elu2(y)
        y = self.dropout2(y)
        y = self.pool2(y)

        y = self.conv3(y)
        y = self.elu3(y)
        y = self.pool3(y)

        B, C, T, F = y.shape
        y = y.reshape(B, C, T * F)

        y = self.tcn(y)
        embeddings = y

        # y2 = y.view(-1, y.shape[2], y.shape[1])
        
        # # y2 = self.linear1(y2)
        # y2 = self.beatthis(y2)
        # combined_features = torch.cat((y, y2), dim=1)
        # y = self.interaction_layer(combined_features)

        y = self.out(y)
        
        
        
        y = self.sigmoid(y)
        #current shape: (1,1, size), target shape( 1, size, 1)
        y = y.squeeze(1)
        y = y.unsqueeze(-1)
        

        if return_embeddings:
            return y, embeddings
        else:
            return y
    
    
def load_checkpoint(model, checkpoint_file, device="cpu"):
    """
    Load model weights from a checkpoint file.

    Args:
        model (nn.Module): PyTorch model
        checkpoint_file (str): Path to checkpoint file
        device (str): 'cpu' or 'cuda'
    """

    checkpoint = torch.load(
        checkpoint_file,
        map_location=device
    )

    # Support both full checkpoints and raw state_dicts
    if isinstance(checkpoint, dict) and "model" in checkpoint:
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()

    return model


if __name__ == "__main__":

    device = "cuda" if torch.cuda.is_available() else "cpu"

    x = torch.rand(1, 1, 6000, 314).to(device)

    model = BeatNet_offlinetcn(downbeats=True)

    checkpoint_file = "/home/nikhil/jukedrummer/offline_tcn_vocals_csv"

    model = load_checkpoint(model, checkpoint_file, device)

    with torch.no_grad():
        embeddings, output = model(x, return_embeddings=True)

    print(embeddings.shape, output.shape)
