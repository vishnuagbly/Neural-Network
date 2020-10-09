#include "Layer.h"

using namespace Eigen;
using namespace std;
using namespace Layers;

Layer::Layer(int units) { this->units = units; }

int Layer::size() const { return units; }

const string DenseLayer::name = "Dense Layer";

string DenseLayer::getName() const { return "Dense Layer"; }

DenseLayer::DenseLayer(const DenseLayer* denseLayer)
    : Layer(denseLayer->size()) {
  this->inputSize = denseLayer->getInputSize();
  this->weights = denseLayer->getWeights();
  this->activation = denseLayer->getActivationObj();
  this->initializer = denseLayer->getInitializerObj();
}

DenseLayer::DenseLayer(int units, MatrixXd weights,
                       const Activation& activation)
    : Layer(units) {
  this->weights = weights;
  this->activation = activation.clone();
}

DenseLayer::DenseLayer(int units, int inputSize, MatrixXd weights,
                       const Activation& activation)
    : Layer(units) {
  this->activation = activation.clone();
  initializeWeights(inputSize, weights);
}

DenseLayer::DenseLayer(int units, const KernelInitializer& initializer,
                       const Activation& activation)
    : Layer(units) {
  this->initializer = initializer.clone();
  this->activation = activation.clone();
}

// basically initialize weights
void DenseLayer::setInputSize(int inputSize) { initializeWeights(inputSize); }

void DenseLayer::initializeWeights(int inputSize) {
  this->inputSize = inputSize;
  if (weights.size() == (inputSize + 1) * size()) return;
  if (initializer.get() == nullptr)
    throw invalid_argument(
        "No initializer exists for this layer, probably wrong inputSize");
  weights = initializer->generate(inputSize, size());
}

void DenseLayer::initializeWeights(int inputSize, MatrixXd weights) {
  if (weights.rows() != size())
    throw invalid_argument("total rows in weights matrix are not correct");
  if (weights.cols() != inputSize + 1)
    throw invalid_argument("total cols in weights matrix are not correct");
  this->inputSize = inputSize;
  this->weights = weights;
}

void DenseLayer::updateWeights(MatrixXd updatedWeights) {
  if (updatedWeights.rows() != weights.rows())
    throw invalid_argument("updated weights rows does not match");
  if (updatedWeights.cols() != weights.cols())
    throw invalid_argument("updated weights cols does not match");
  this->weights = updatedWeights;
}

MatrixXd DenseLayer::getWeights() const { return weights; }

Activation* DenseLayer::getActivation() { return this->activation.get(); }

int DenseLayer::getInputSize() const { return this->inputSize; }

unique_ptr<Activation> DenseLayer::getActivationObj() const {
  return activation->clone();
}

unique_ptr<KernelInitializer> DenseLayer::getInitializerObj() const {
  return initializer->clone();
}

unique_ptr<Layer> DenseLayer::getLayer() const {
  return make_unique<DenseLayer>(this);
}

const string InputLayer::name = "Input Layer";

string InputLayer::getName() const { return InputLayer::name; }

void InputLayer::setInputSize(int inputSize) { initializeWeights(inputSize); }

void InputLayer::initializeWeights(int inputSize) {
  throw runtime_error("input layer doesn't have weights to initialize");
}

void InputLayer::initializeWeights(int inputSize, MatrixXd weights) {
  throw runtime_error("input layer doesn't have weights to initialize");
}

InputLayer::InputLayer(int size) : Layer(size) {}

void InputLayer::updateWeights(MatrixXd updatedWeights) {
  throw runtime_error("Input Layer doesn't have weights to update");
}

MatrixXd InputLayer::getWeights() const {
  throw runtime_error("Input Layer doesn't have weights");
}

Activation* InputLayer::getActivation() {
  throw runtime_error("Input Layer doesn't have activations");
}

unique_ptr<Layer> InputLayer::getLayer() const {
  return make_unique<InputLayer>(size());
}
