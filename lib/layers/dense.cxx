#include "Layer.h"
using namespace std;
using namespace Eigen;

const string Layers::Dense::name = "Dense Layer";

string Layers::Dense::getName() const { return "Dense Layer"; }

Layers::Dense::Dense(const Dense* denseLayer) : Layer(denseLayer->size()) {
  this->inputSize = denseLayer->getInputSize();
  this->weights = denseLayer->getWeights();
  this->activation = denseLayer->getActivationObj();
  this->initializer = denseLayer->getInitializerObj();
}

Layers::Dense::Dense(int units, MatrixXd weights, const Activation& activation)
    : Layer(units) {
  this->weights = weights;
  this->activation = activation.clone();
}

Layers::Dense::Dense(int units, int inputSize, MatrixXd weights,
                     const Activation& activation)
    : Layer(units) {
  this->activation = activation.clone();
  initializeWeights(inputSize, weights);
}

Layers::Dense::Dense(int units, const KernelInitializer& initializer,
                     const Activation& activation)
    : Layer(units) {
  this->initializer = initializer.clone();
  this->activation = activation.clone();
}

// basically initialize weights
void Layers::Dense::setInputSize(int inputSize) {
  initializeWeights(inputSize);
}

void Layers::Dense::initializeWeights(int inputSize) {
  this->inputSize = inputSize;
  if (weights.size() == (inputSize + 1) * units) return;
  if (initializer.get() == nullptr)
    throw invalid_argument(
        "No initializer exists for this layer, probably wrong inputSize");
  weights = initializer->generate(inputSize, units);
}

void Layers::Dense::initializeWeights(int inputSize, const MatrixXd& weights) {
  if (weights.rows() != units)
    throw invalid_argument("total rows in weights matrix are not correct");
  if (weights.cols() != inputSize + 1)
    throw invalid_argument("total cols in weights matrix are not correct");
  this->inputSize = inputSize;
  this->weights = weights;
}

void Layers::Dense::updateWeights(const MatrixXd& updatedWeights) {
  if (updatedWeights.rows() != weights.rows())
    throw invalid_argument("updated weights rows does not match");
  if (updatedWeights.cols() != weights.cols())
    throw invalid_argument("updated weights cols does not match");
  this->weights = updatedWeights;
}

VectorXd Layers::Dense::getOutputValues(const VectorXd& lastLayer) const {
  return activation->actFn(getZValues(lastLayer));
}

VectorXd Layers::Dense::getZValues(const VectorXd& lastLayer) const {
  if (lastLayer.size() != weights.cols() - 1)
    throw invalid_argument("can't get z values, last layer is of wrong size\n");
  VectorXd tempLastLayer(lastLayer.size() + 1);
  tempLastLayer << 1, lastLayer;
  return weights * tempLastLayer;
}

VectorXd Layers::Dense::backPropagateDell(const VectorXd& thisLayerDell) const {
  if (thisLayerDell.size() != units)
    throw invalid_argument("this layer dell vector is of wrong size\n");
  auto dell = weights.transpose() * thisLayerDell;
  return dell.block(1, 0, dell.size() - 1, 1);
}

MatrixXd Layers::Dense::calcBigDell(const VectorXd& thisLayerDell,
                                    const VectorXd& lastLayer,
                                    const double lambda) const {
  if (thisLayerDell.size() != units)
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

MatrixXd Layers::Dense::getWeights() const { return weights; }

Activation* Layers::Dense::getActivation() const {
  return this->activation.get();
}

int Layers::Dense::getInputSize() const { return this->inputSize; }

unique_ptr<Activation> Layers::Dense::getActivationObj() const {
  return activation->clone();
}

unique_ptr<KernelInitializer> Layers::Dense::getInitializerObj() const {
  return initializer->clone();
}

unique_ptr<Layer> Layers::Dense::clone() const {
  return make_unique<Dense>(this);
}