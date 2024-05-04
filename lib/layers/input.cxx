#include "Layer.h"
using namespace Eigen;
using namespace std;

const string Layers::Input::name = "Input Layer";

string Layers::Input::getName() const { return Layers::Input::name; }

void Layers::Input::setInputSize(int inputSize) {
  initializeWeights(inputSize);
}

void Layers::Input::initializeWeights(int inputSize) {
  throw runtime_error("input layer doesn't have weights to initialize");
}

void Layers::Input::initializeWeights(int inputSize, const MatrixXd& weights) {
  throw runtime_error("input layer doesn't have weights to initialize");
}

Layers::Input::Input(int size) : Layer(size) {}

void Layers::Input::updateWeights(const MatrixXd& updatedWeights) {
  throw runtime_error("Input Layer doesn't have weights to update");
}

VectorXd Layers::Input::getZValues(const VectorXd& lastLayer) const {
  throw runtime_error("Input Layer doesn't have Z values");
}

VectorXd Layers::Input::getOutputValues(const VectorXd& lastLayer) const {
  throw runtime_error("Input Layer doesn't have output values\n");
}

VectorXd Layers::Input::backPropagateDell(const VectorXd& thisLayerDell) const {
  throw runtime_error("Input Layer cannot backpropagate\n");
}

MatrixXd Layers::Input::calcBigDell(const VectorXd& thisLayerDell,
                                    const VectorXd& lastLayer,
                                    const double lambda) const {
  throw runtime_error("Input layer cannot calc Big dell values\n");
}

MatrixXd Layers::Input::getWeights() const {
  throw runtime_error("Input Layer doesn't have weights");
}

Activation* Layers::Input::getActivation() const {
  throw runtime_error("Input Layer doesn't have activations");
}

unique_ptr<Layer> Layers::Input::clone() const {
  return make_unique<Input>(size());
}
