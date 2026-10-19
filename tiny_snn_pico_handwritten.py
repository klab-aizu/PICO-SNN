from random import random
from PIL import Image
import numpy as np
import sys
import matplotlib.pyplot as plt

N_TIMESTEPS = 500
N_TRAIN_SET = 10
N_INPUTS = 28*28
N_NEURONS = 10
W_MAX = 1.39
W_MIN = 0
W_INHIBITORY = -140*1.5
REFRAC_PERIOD = 20
LEARNING_RATIO = 0.01

def spike(p: float)->int:
  """
  args:
    p (float): probability of spike (normalized)
  returns:
    0 (no spike) or 1 (spike)
  """
  r = random()
  if r < p:
    return 1
  return 0

def generate_spiketrains(pattern: np.ndarray) -> np.ndarray:
  """
  args:

    pattern (np.ndarray): 1D array (i.e [1, 0])
  returns:
    spiketrains (np.ndarray): 2D array containing the corresponding spiketrains
  """
  spiketrain = np.full((pattern.shape[0], N_TIMESTEPS), 0)
  for i, val in enumerate(pattern):
    for t in range(N_TIMESTEPS):
        spiketrain[i][t] = spike(val)
  return spiketrain

def assign_lables(snn, train_set: np.ndarray, train_set_labels: np.ndarray)->np.ndarray:
    """
    args:
        snn (class '__main__.SNN'): instantiated snn
        train_set (np.ndarray): training dataset
        train_set_labels (np.ndarray): labels for training dataset (corresponding in positions)
    returns:
        assignments (np.ndarray): label assignments for each neurons (corresponding in positions)
    """
    assignments = np.full((N_TRAIN_SET, N_NEURONS), -1)
    for i, pattern in enumerate(train_set):
        spiketrains = snn.forward(generate_spiketrains(pattern), train=False)
        assignments[i] = np.sum(spiketrains, axis=1, keepdims=False)
    assignments = assignments.transpose()
    assignments = np.argmax(assignments, axis=1)
    assignments = labels_train_set[assignments]

    return assignments

class LIF():
    """
    Leaky-Integrate-and-Fire neuron
    """
    def __init__(self)->None:
        """
        attributes:
            self.leakage (float): constant membrane leakage 
            self.threshold (float): firing threshold
            self.membrane_potential (float): membrane potential
            self.refrac_count (int): remaining refractory period
        """
        self.leakage = 0.01
        self.threshold = N_INPUTS / 10
        self.membrane_potential = 0.0
        self.refrac_count = 0

    def forward(self, z: float)->int:
        """
        args:
            z (float): input current
        returns:
            (float): output spike (0 or 1)
        """
        if self.refrac_count > 0:
            self.refrac_count -= 1
            return 0

        self.membrane_potential = self.membrane_potential + z
        if self.membrane_potential >= self.threshold:
            self.membrane_potential = 0
            return 1
        else:
            self.membrane_potential = self.membrane_potential - self.leakage
            return 0

class SNN():
    """
    Spiking Neural Network
    """
    def __init__(self, N_INPUTS: int, N_NEURONS: int)->None:
        """
        Attributes:
            self.N_INPUTS (int): number of input neurons
            self.N_NEURONS (int): number of excitatory neurons
            self.weights (np.ndarray): weights metrix (N_NEURONS, N_INPUTS)
            self.neurons (list): list of LIF neurons
        """
        self.N_INPUTS = N_INPUTS
        self.N_NEURONS = N_NEURONS
        self.weights = np.array([[random() for c in range(N_INPUTS)] for r in range(N_NEURONS)])
        self.neurons = [LIF() for n in range(N_NEURONS)]
        self.normalize()

    def normalize(self)->None:
        """
        Normalize weights (sum of weight value per neuron becomes 1)
        """
        self.weights = self.weights.clip(W_MIN, W_MAX)
        self.weights /= np.sum(self.weights, axis=1, keepdims=True)
        self.weights *= (N_INPUTS / 10)

    def inhibitory(self, fired_indx: int)->None:
        """
        args:
            fired_indx (int): index of the fired neuron
        returns:
            None
        """
        for indx, neuron in enumerate(self.neurons):
            if indx != fired_indx:
                neuron.membrane_potential += W_INHIBITORY

    def refractory(self, fired_indx: int)->None:
        """
        args:
            fired_indx (int): index of the fired neuron
        returns:
            None
        """
        self.neurons[fired_indx].refrac_count = REFRAC_PERIOD

    def forward(self, spiketrain: np.ndarray, train: bool)->np.ndarray:
        """
        args:
            spiketrain (np.ndarray): (n_input x N_TIMESTEPS) array
            train (bool): training on input (true), inference on input (false)
        returns:
            output_spikes (np.ndarray): (N_NEURONS x N_TIMESTEPS) array 
        """
        output_spikes = np.full((self.N_NEURONS, N_TIMESTEPS), 0)

        for i in range(5, N_TIMESTEPS):
            inputs = np.dot(self.weights, spiketrain[:, i])

            for j, neuron in enumerate(self.neurons):
                output_spikes[j][i] = neuron.forward(inputs[j])

                if output_spikes[j][i] == 1:
                    if train:
                        for k in range(spiketrain.shape[0]):
                            if 1 in spiketrain[k][i-5:i]:
                                self.weights[j][k] *= (1+LEARNING_RATIO)
                            elif 1 in spiketrain[k][i:i+5]:
                                self.weights[j][k] *= (1-LEARNING_RATIO)
                        self.normalize()
                    self.inhibitory(j)
                    self.refractory(j)
        return output_spikes

def image2bin(path: str, dim: tuple, threshold: int)->np.ndarray:
    """
    args:
        path (str): path to image file
        dim: (tuple): dimension of converted image
        threshold (int): 1 for <threshold, otherwise 0
    returns:
        img (np.ndarray): converted image
    """
    img = Image.open(path).convert('L')
    img = img.resize(dim, Image.LANCZOS)
    img = np.array(img)
    img = np.where(img<threshold, 1, 0)
    return img

img_0 = image2bin('img/0.jpeg', (28, 28), 100)
img_1 = image2bin('img/1.jpeg', (28, 28), 100)
img_2 = image2bin('img/2.jpeg', (28, 28), 100)
img_3 = image2bin('img/3.jpeg', (28, 28), 100)
img_4 = image2bin('img/4.jpeg', (28, 28), 100)
img_5 = image2bin('img/5.jpeg', (28, 28), 100)
img_6 = image2bin('img/6.jpeg', (28, 28), 100)
img_7 = image2bin('img/7.jpeg', (28, 28), 100)
img_8 = image2bin('img/8.jpeg', (28, 28), 100)
img_9 = image2bin('img/9.jpeg', (28, 28), 100)

train_set = np.stack((img_0, img_1, img_2, img_3, img_4, img_5, img_6, img_7, img_8, img_9), axis=0)

train_set = train_set.reshape(train_set.shape[0], -1)
np.set_printoptions(linewidth=np.inf, threshold=sys.maxsize)
labels_train_set = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])
test_set = train_set
labels_test_set = labels_train_set

snn = SNN(N_INPUTS, N_NEURONS)

for epoch in range(50):
  print(f"Epoch {epoch}")
  for pattern in train_set:
    spiketrains = snn.forward(generate_spiketrains(pattern), train=True)
    snn.neurons[np.argmax(np.sum(spiketrains, axis=1))].refrac_count = 500*3
  if epoch == 49:
    fig, ax = plt.subplots(1, N_NEURONS)
    for i in range(N_NEURONS):
        ax[i].imshow(snn.weights[i].reshape(28, 28))
    fig.show()
    fig.savefig('recep.png')
    input()
print(f"Trained weights: {snn.weights}")



assignments = assign_lables(snn, train_set, labels_train_set)

correct = 0
for i, pattern in enumerate(test_set):
    print(pattern)
    spiketrains = snn.forward(generate_spiketrains(pattern), train=False)
    reacted_neuron = np.argmax(np.sum(spiketrains, axis=1, keepdims=True))
    print(f"prediction: {assignments[reacted_neuron]}")
    if assignments[reacted_neuron] == labels_test_set[i]:
        print("correct")
        correct += 1
    else:
        print("wrong")
    print()

print(f"Accuracy: {correct/len(test_set)*100} %")
