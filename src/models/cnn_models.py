from functools import reduce
import operator

import torch
import torch.nn as nn


def conv_relu_bn(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=1, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]


def conv_down(cin, cout):
    return [
        nn.Conv2d(cin, cout, kernel_size=3, stride=2, padding=1),
        nn.ReLU(),
        nn.BatchNorm2d(cout),
    ]

def out_layers(num_features, hidden_size, num_classes):
    return [
    nn.Flatten(start_dim=1),
    nn.Linear(num_features, hidden_size),
    nn.ReLU(),
    # nn.Dropout(0.3),
    nn.Linear(hidden_size, num_classes)
    ]


def VanillaCNN(cfg, input_size, num_classes):
    layers = []
    cin = input_size[0]
    cout = 16
    for i in range(cfg["num_layers"]):
        layers.extend(conv_relu_bn(cin, cout))
        layers.extend(conv_relu_bn(cout, cout))
        layers.extend(conv_down(cout, 2 * cout))
        cin = 2 * cout
        cout = 2 * cout
    conv_model = nn.Sequential(*layers)

    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
    out_layers = [nn.Flatten(start_dim=1), nn.Linear(num_features, num_classes)]
    return nn.Sequential(conv_model, *out_layers)


def SimpleCNN(cfg, input_size, num_classes):
    layers = []
    cin = input_size[0]
    cout = cfg["cout"]
    hidden_size = cfg["hidden_size"]
    for i in range(cfg["num_layers"]):
        layers.extend(conv_relu_bn(cin, cout))
        layers.extend(conv_relu_bn(cout, cout))
        layers.extend(conv_down(cout, 2 * cout))
        cin = 2 * cout
        cout = 2 * cout
    conv_model = nn.Sequential(*layers)

    # Compute the output size of the convolutional part
    probing_tensor = torch.zeros((1,) + input_size)
    out_cnn = conv_model(probing_tensor)  # B, K, H, W
    num_features = reduce(operator.mul, out_cnn.shape[1:], 1)
    return nn.Sequential(conv_model, *out_layers(num_features, hidden_size, num_classes))