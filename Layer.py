import numpy as np
import math


# Dense layer
class Layer:
    # Layer initialization
    def __init__(self, n_inputs, n_neurons):
        # Initialize weights and biases
        self.weights = 0.1 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))

    # Forward pass
    def forward(self, inputs):
        # Calculate output values from inputs, weights and biases
        self.output = np.dot(inputs, self.weights) + self.biases

    def ReLU(self):
        # Apply ReLU withot saving
        return np.maximum(0, self.output)

    def ApplyReLU(self):
        # Apply ReLU and save the results in output
        self.output = np.maximum(0, self.output)

    # I stole this
    def SoftMax(self):
        exp_values = np.exp(self.output)
        s = np.sum(exp_values)
        exp_values_ratio = exp_values / s
        return exp_values_ratio

    # I made this
    def ApplySoftMax(self):
        exp_values = np.exp(self.output)
        s = np.sum(exp_values)
        exp_values_ratio = exp_values / s
        self.output = np.array(exp_values_ratio)

    def CalulateLoss(self, class_targets):
        #loss = - (math.log(np.dot(self.output, target_output)))
        # You can use this instead
        #loss = - math.log(softmax_output[np.argmax(target_output)])
        tempOutput = self.output
        # clip, idk why
        tempOutput = np.clip(tempOutput, 1e-7, 1 - 1e-7)
        avg_loss =  np.mean(-np.log(np.array(tempOutput)[range(len(tempOutput)), class_targets]))
        return avg_loss
    def CalulateAccuracy(self, class_targets):
        # Get the predection of the output layer as indices
        predections = np.argmax(self.output, axis= 1)
        # Compare between the lables and predectied lables the higher the better
        numberOfCorrect = np.count_nonzero(predections == class_targets)
        # cope
        return numberOfCorrect /len(class_targets)