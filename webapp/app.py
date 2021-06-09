#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
###############################################################################
# app.py
#
# Revision:     1.00
# Date:         6/8/2021
# Author:       Alex
#
# Purpose:      Displays a webpage using Flask.  Loads the GAN generator
#               PyTorch model and displays a random generated image.
#
# Notes:
# 1. If you've trained another generator change the GEN_FN variable to point
#    to its location.
#
###############################################################################
"""

import math
import os
from datetime import datetime
from flask import Flask
from flask import render_template
import torch
import torch.nn as nn
from PIL import Image
import numpy as np


# %% Classes
class Generator(nn.Module):
    """Generator class defining the neural network architecture."""
    def __init__(self, ngpu):
        super().__init__()
        self.ngpu = ngpu
        assert math.log(NGF, 2).is_integer(), \
            "Ensure image dimensions are a power of two!"
        self.num_blocks = int(math.log(NGF, 2) - 1)
        self.main = nn.Sequential(
            self._build_body(),
            nn.ConvTranspose2d(NGF, NC, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def _build_body(self):
        """Create all layers of the Generator excluding the output layers."""
        # Powers of two in reverse order
        channels = [NGF*(2**x) for x in range(self.num_blocks-2,-1,-1)]
        input_layer = self._convtrans_block(NZ, channels[0], kernel_size=4,
                                            stride=1, padding=0, bias=False)
        body_layers = [self._convtrans_block(inp_chan, out_chan, kernel_size=4,
                                             stride=2, padding=1, bias=False)
                       for inp_chan, out_chan in zip(channels, channels[1:])]
        return nn.Sequential(input_layer, *body_layers)

    def _convtrans_block(self, in_chan, out_chan, *args, **kwargs):
        """Returns a unit block of layers used in the body of the Generator neural network."""
        return nn.Sequential(
            nn.ConvTranspose2d(in_chan, out_chan, *args, **kwargs),
            nn.BatchNorm2d(out_chan),
            nn.ReLU(True)
    )

    def forward(self, input_data):
        """Called during forward pass through neural network."""
        return self.main(input_data)


# %% Functions
def generate_new_image():
    """Generates a new image and saves it to disk with a timestamp so that the
    filename is unique.  This prevents browser caching of images.  Returns the
    generated image's filename so that it can be passed to the HTML template.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d_%H:%M:%S')
    noise = torch.randn(1, NZ, 1, 1, device=device) # Latent Z vector
    with torch.no_grad():
        generated_img = model(noise).detach().cpu().numpy()
    # Rearrange axes so channels are last, and scale values from 0 to 255
    rearranged_arr = np.moveaxis(generated_img.squeeze(), (0,1,2), (2,0,1))
    scaled_arr = (rearranged_arr * 255 / np.max(rearranged_arr)).astype('uint8')
    # Create RGB image from Numpy array and save to disk with unique filename.
    pil_img = Image.fromarray(scaled_arr, "RGB")
    filename = f'static/img/pokegan_{timestamp}.png' # Avoid browser caching
    pil_img.convert("RGB").save(filename)
    return filename


# %% Define Parameters
PATH = 'static/img/'  # Path to folder containing website images

# PyTorch parameters
NC = 3     # Number of channels in the training images. For color images this is 3
NZ = 100   # Size of z latent vector (i.e. size of generator input)
NGF = 128  # Size of feature maps in generator
NGPU = 1   # Number of GPUs available. Use 0 for CPU mode.


# %% Load Generator Model
GEN_FN = '../models/Generator_2021-06-09_16:28:18.pth'
model = Generator(NGPU)
model.load_state_dict(torch.load(GEN_FN))
model.eval()
device = torch.device("cuda:0" if (torch.cuda.is_available() and NGPU > 0) \
                      else "cpu")
model.to(device)


# %% Run Flask Application
app = Flask(__name__)

@app.route("/")
def pokegan():
    """Generates a new image and refreshes the webpage whenever user
    queries server.  Old generated images are deleted, and the new image's
    filename is passed to the HTML template.
    """
    for file in os.listdir(PATH):
        if file.startswith("pokegan_20"): # Delete old generated images
            os.remove(PATH + file)        # to prevent clutter
    filename = generate_new_image()
    return render_template('index.html', filename=filename)


if __name__ == '__main__':
    generate_new_image() # For debugging

