# Neural-Network
This is a super easy to use Neural Network Library. It can be used to make multiple layered Neural Networks, with lots of customization.

# Requirements
Eigen library should be already installed and available, since this library uses Eigen v3.5.8 library.

# How To Use
To use this Library we need to add following flags:-

- -I "path/to/this/Library/headers"

- -o "path/to/this/Library/lib/*.cxx" "path/to/this/Library/lib/layers/*.cxx"

# Documentation
Please refer to Report.pdf for full documentation.

# Example
```
#include <NeuralNetwork.h>

int main() {
  auto nn = NeuralNetwork();
  nn.add(Layers::Input(4));
  nn.add(Layers::Dense(6, Activations::Relu()));
  nn.add(Layers::Dense(7));
  
  MatrixXd inputs = getInputs();
  MatrixXd expectedOutputs = getOutputs();
  
  int batchSize = 20, numOfEpochs = 5;
  
  nn.train(inputs, expectedOutputs, batchSize, numOfEpochs);
  nn.printResults(inputs, expectedOutputs);
  
  return 0;
}
```
