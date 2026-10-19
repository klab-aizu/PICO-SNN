from random import random
import numpy as np
import pandas as pd

n_timesteps = 500
n_train_set = 105 # needs to be multiple of 3
n_inputs = 4
n_neurons = 30

def load_iris(path: str)->np.ndarray:
    """
    args:
        path (str): path to iris data csv
    returns:
        train_set (np.ndarray): 2D array of iris training data
        labels_train_set (np.ndarray): 1D array of training labels
        test_set (np.ndarray): 2D array of iris test data
        labels_test_set (np.ndarray): 1D array of test labels
    """
    df = pd.read_csv(path)
    df = df.iloc[:, 1:].to_numpy()
    df[:, :4] /= np.max(df[:, :4], axis=0, keepdims=True)

    n_train = int(n_train_set / 3)

    train_set = np.concatenate((df[:n_train, :4], df[50: 50+n_train, :4], df[100: 100+n_train, :4]), axis=0)
    labels_train_set = np.concatenate((df[:n_train, 4], df[50: 50+n_train, 4], df[100: 100+n_train, 4]), axis=0)

    test_set = np.concatenate((df[n_train: 50, :4], df[50+n_train: 100, :4], df[100+n_train:, :4]), axis=0)
    labels_test_set = np.concatenate((df[n_train:50, 4], df[50+n_train: 100, 4], df[100+n_train:, 4]), axis=0)

    return train_set, labels_train_set, test_set, labels_test_set

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
  spiketrain = np.full((pattern.shape[0], n_timesteps), 0)
  for i, val in enumerate(pattern):
    for t in range(n_timesteps):
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
    assignments = np.full((n_train_set, n_neurons), -1)
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
        self.threshold = 1.0
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
    def __init__(self, n_inputs: int, n_neurons: int)->None:
        """
        Attributes:
            self.n_inputs (int): number of input neurons
            self.n_neurons (int): number of excitatory neurons
            self.weights (np.ndarray): weights metrix (n_neurons, n_inputs)
            self.neurons (list): list of LIF neurons
        """
        self.n_inputs = n_inputs
        self.n_neurons = n_neurons
        self.weights = np.array([[random() for c in range(n_inputs)] for r in range(n_neurons)])
        self.neurons = [LIF() for n in range(n_neurons)]
        self.normalize()

    def normalize(self)->None:
        """
        Normalize weights (sum of weight value per neuron becomes 1)
        """
        self.weights /= np.sum(self.weights, axis=1, keepdims=True)
        self.weights *= (n_inputs / 10)

    def inhibitory(self, fired_indx: int)->None:
        """
        args:
            fired_indx (int): index of the fired neuron
        returns:
            None
        """
        for indx, neuron in enumerate(self.neurons):
            if indx != fired_indx:
                neuron.membrane_potential -= 2

    def refractory(self, fired_indx: int)->None:
        """
        args:
            fired_indx (int): index of the fired neuron
        returns:
            None
        """
        self.neurons[fired_indx].refrac_count = 20

    def forward(self, spiketrain: np.ndarray, train: bool)->np.ndarray:
        """
        args:
            spiketrain (np.ndarray): (n_input x n_timesteps) array
            train (bool): training on input (true), inference on input (false)
        returns:
            output_spikes (np.ndarray): (n_neurons x n_timesteps) array 
        """
        output_spikes = np.full((self.n_neurons, n_timesteps), 0)

        for i in range(5, n_timesteps):
            inputs = np.dot(self.weights, spiketrain[:, i])

            for j, neuron in enumerate(self.neurons):
                output_spikes[j][i] = neuron.forward(inputs[j])

                if output_spikes[j][i] == 1:

                    if train:
                        for k in range(spiketrain.shape[0]):
                            if 1 in spiketrain[k][i-5:i]:
                                self.weights[j][k] *= 1.1
                            elif 1 in spiketrain[k][i:i+5]:
                                self.weights[j][k] *= 0.9
                        self.normalize()
                    self.inhibitory(j)
                    self.refractory(j)
        return output_spikes

train_set, labels_train_set, test_set, labels_test_set = load_iris("Iris.csv")

snn = SNN(n_inputs, n_neurons)

for epoch in range(20):
  print(f"Epoch {epoch}")
  for pattern in train_set:
    spiketrains = snn.forward(generate_spiketrains(pattern), train=True)

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