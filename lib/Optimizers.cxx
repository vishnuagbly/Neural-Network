#include "Optimizer.h"
using namespace std;
using namespace Eigen;

Optimizers::SGD::SGD(double learningRate, double momentum) {
  if (momentum < 0 || momentum >= 1)
    throw invalid_argument("momentum values must belong to [0, 1) range\n");
  this->learningRate = learningRate;
  this->momentum = momentum;
}

Optimizers::SGD::SGD(const SGD& obj) {
  this->learningRate = obj.learningRate;
  this->momentum = obj.momentum;
  this->velocity = obj.velocity;
}

vector<MatrixXd> Optimizers::SGD::applyOptimzer(vector<MatrixXd> values,
                                                vector<MatrixXd> grads) {
  if (values.size() != grads.size())
    throw invalid_argument("values and grads of different size\n");
  bool firstTime = (velocity.size() != grads.size());
  for (int i = 0; i < values.size(); i++) {
    if (firstTime)
      velocity.emplace_back(MatrixXd::Zero(grads[i].rows(), grads[i].cols()));
    velocity[i] = (momentum * velocity[i]) + ((1 - momentum) * (grads[i]));
    values[i] = values[i] - (learningRate * velocity[i]);
  }
  return values;
}

void Optimizers::SGD::clear() { velocity.clear(); }

unique_ptr<Optimizer> Optimizers::SGD::clone() const {
  return make_unique<SGD>(*this);
}

Optimizers::Adam::Adam(double alpha, double beta1, double beta2) {
  if (beta1 < 0 || beta1 >= 1)
    throw invalid_argument("beta1 values must belong to [0, 1) range\n");
  this->alpha = alpha;
  this->beta1 = beta1;
  this->beta2 = beta2;
}

Optimizers::Adam::Adam(const Adam& obj) {
  this->alpha = obj.alpha;
  this->beta1 = obj.beta1;
  this->beta2 = obj.beta2;
  this->moment1 = obj.moment1;
  this->moment2 = obj.moment2;
}

vector<MatrixXd> Optimizers::Adam::applyOptimzer(vector<MatrixXd> values,
                                                 vector<MatrixXd> grads) {
  if (values.size() != grads.size())
    throw invalid_argument("values and grads of different size\n");
  bool firstTime = (moment1.size() != grads.size());
  for (int i = 0; i < values.size(); i++) {
    if (firstTime) {
      moment1.emplace_back(MatrixXd::Zero(grads[i].rows(), grads[i].cols()));
      moment2.emplace_back(MatrixXd::Zero(grads[i].rows(), grads[i].cols()));
    }
    moment1[i] = (beta1 * moment1[i]) + ((1 - beta1) * (grads[i]));
    moment2[i] = (beta2 * moment2[i]) +
                 ((1 - beta2) * (grads[i].cwiseProduct(grads[i])));
    MatrixXd moment1Dash = moment1[i] / (1 - beta1);
    MatrixXd moment2Dash = moment2[i] / (1 - beta2);
    values[i] =
        values[i] -
        (alpha *
         (moment1Dash.cwiseQuotient(
             moment2Dash.cwiseSqrt() +
             MatrixXd::Constant(grads[i].rows(), grads[i].cols(), epsilon))));
  }
  return values;
}

void Optimizers::Adam::clear() {
  moment1.clear();
  moment2.clear();
}

unique_ptr<Optimizer> Optimizers::Adam::clone() const {
  return make_unique<Adam>(*this);
}