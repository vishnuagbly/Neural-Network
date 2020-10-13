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

VectorXd DenseLayer::getOutputValues(const VectorXd& lastLayer) {
  return activation->actFn(getZValues(lastLayer));
}

VectorXd DenseLayer::getZValues(const VectorXd& lastLayer) {
  if (lastLayer.size() != weights.cols() - 1)
    throw invalid_argument("can't get z values, last layer is of wrong size\n");
  VectorXd tempLastLayer(lastLayer.size() + 1);
  tempLastLayer << 1, lastLayer;
  return weights * tempLastLayer;
}

VectorXd DenseLayer::backPropagateDell(const VectorXd& thisLayerDell) {
  if (thisLayerDell.size() != size())
    throw invalid_argument("this layer dell vector is of wrong size\n");
  auto dell = weights.transpose() * thisLayerDell;
  return dell.block(1, 0, dell.size() - 1, 1);
}

MatrixXd DenseLayer::calcBigDell(const VectorXd& thisLayerDell,
                                 const VectorXd& lastLayer,
                                 const double lambda) {
  if (thisLayerDell.size() != size())
    throw invalid_argument("this Layer dell vector is of wrong size\n");
  if (lastLayer.size() != weights.cols() - 1)
    throw invalid_argument("last layer size is of wrong size\n");

  VectorXd tempLastLayer(weights.cols());
  tempLastLayer << 1, lastLayer;
  auto tempBigDell = thisLayerDell * tempLastLayer.transpose();
  MatrixXd bigDell(weights.rows(), weights.cols());
  bigDell << getBias(tempBigDell),
      getKernel(tempBigDell) + (lambda * getKernel(weights));
  return bigDell;
}

MatrixXd DenseLayer::getBias(const MatrixXd& matrix) {
  return matrix.block(0, 0, matrix.rows(), 1);
}

MatrixXd DenseLayer::getKernel(const MatrixXd& matrix) {
  return matrix.block(0, 1, matrix.rows(), matrix.cols() - 1);
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

unique_ptr<Layer> DenseLayer::clone() const {
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

VectorXd InputLayer::getZValues(const VectorXd& lastLayer) {
  throw runtime_error("Input Layer doesn't have Z values");
}

VectorXd InputLayer::getOutputValues(const VectorXd& lastLayer) {
  throw runtime_error("Input Layer doesn't have output values\n");
}

VectorXd InputLayer::backPropagateDell(const VectorXd& thisLayerDell) {
  throw runtime_error("Input Layer cannot backpropagate\n");
}

MatrixXd InputLayer::calcBigDell(const VectorXd& thisLayerDell,
                                 const VectorXd& lastLayer,
                                 const double lambda) {
  throw runtime_error("Input layer cannot calc Big dell values\n");
}

MatrixXd InputLayer::getWeights() const {
  throw runtime_error("Input Layer doesn't have weights");
}

Activation* InputLayer::getActivation() {
  throw runtime_error("Input Layer doesn't have activations");
}

unique_ptr<Layer> InputLayer::clone() const {
  return make_unique<InputLayer>(size());
}
