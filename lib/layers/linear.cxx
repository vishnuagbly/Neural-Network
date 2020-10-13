#include "Layer.h"
using namespace std;
using namespace Eigen;

const string Layers::Linear::name = "Linear";

string Layers::Linear::getName() const { return name; }

Layers::Linear::Linear(const Linear* linear) : Layer(linear->size()) {
  this->weights = linear->getWeights();
  this->activation = linear->activation->clone();
  this->initializer = linear->initializer->clone();
}

Layers::Linear::Linear(const Activation& activation,
                       const KernelInitializer& initializer)
    : Layer(0) {
  this->activation = activation.clone();
  this->initializer = initializer.clone();
}

void Layers::Linear::setInputSize(int inputSize) {
  initializeWeights(inputSize);
}

void Layers::Linear::initializeWeights(int inputSize) {
  if (units != 0) throw runtime_error("already initialized layer\n");
  units = inputSize;
  if (weights.size() == 2 * units) return;
  if (initializer.get() == nullptr)
    throw invalid_argument(
        "No initializer exists for this layer, probably wrong inputSize");
  weights = initializer->generate(1, units);
}

void Layers::Linear::initializeWeights(int inputSize, const MatrixXd& weights) {
  if (weights.rows() != units)
    throw invalid_argument("total rows in weights matrix are not correct");
  if (weights.cols() != 2)
    throw invalid_argument("total cols in weights matrix are not correct");
  this->weights = weights;
}

void Layers::Linear::updateWeights(const MatrixXd& updatedWeights) {
  if (updatedWeights.rows() != weights.rows())
    throw invalid_argument("updated weights rows does not match");
  if (updatedWeights.cols() != weights.cols())
    throw invalid_argument("updated weights cols does not match");
  this->weights = updatedWeights;
}

VectorXd Layers::Linear::getOutputValues(const VectorXd& lastLayer) const {
  return activation->actFn(getZValues(lastLayer));
}

VectorXd Layers::Linear::getZValues(const VectorXd& lastLayer) const {
  if (lastLayer.size() != units)
    throw invalid_argument("can't get z values, last layer is of wrong size\n");
  return weights.col(0) + weights.col(1).cwiseProduct(lastLayer);
}

VectorXd Layers::Linear::backPropagateDell(
    const VectorXd& thisLayerDell) const {
  if (thisLayerDell.size() != units)
    throw invalid_argument("this layer dell vector is of wrong size\n");
  return weights.col(1).cwiseProduct(thisLayerDell);
}

MatrixXd Layers::Linear::calcBigDell(const VectorXd& thisLayerDell,
                                     const VectorXd& lastLayer,
                                     const double lambda) const {
  if (thisLayerDell.size() != units)
    throw invalid_argument("this Layer dell vector is of wrong size\n");
  if (lastLayer.size() != units)
    throw invalid_argument("last layer size is of wrong size\n");

  MatrixXd tempBigDell(weights.rows(), weights.cols());
  tempBigDell << thisLayerDell, thisLayerDell.cwiseProduct(lastLayer);
  MatrixXd bigDell(weights.rows(), weights.cols());
  bigDell << tempBigDell.col(0), tempBigDell.col(1) + (lambda * weights.col(1));
  return bigDell;
}

MatrixXd Layers::Linear::getWeights() const { return weights; }

Activation* Layers::Linear::getActivation() const {
  return this->activation.get();
}

unique_ptr<Layer> Layers::Linear::clone() const {
  return make_unique<Linear>(this);
}

const string Layers::ConstLinear::name = "Const Linear";

void Layers::ConstLinear::updateWeights(const MatrixXd& updatedWeights) {
  if (updatedWeights.rows() != getWeights().rows())
    throw invalid_argument("updated weights rows does not match");
  if (updatedWeights.cols() != getWeights().cols())
    throw invalid_argument("updated weights cols does not match");

  // Const Linear cannot update weights;
}