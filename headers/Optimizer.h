#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include <Eigen/Dense>
#include <memory>
#include <vector>

class Optimizer {
 public:
  virtual std::vector<Eigen::MatrixXd> applyOptimzer(
      std::vector<Eigen::MatrixXd> values,
      std::vector<Eigen::MatrixXd> grads) = 0;
  virtual void clear() = 0;
  virtual std::unique_ptr<Optimizer> clone() const = 0;
};

namespace Optimizers {
/// SGD assumes std::vector<Eigen::MatrixXd> type can perform basic arithemetic
/// operations.
class SGD : public Optimizer {
  double learningRate;
  double momentum;
  std::vector<Eigen::MatrixXd> velocity;

 public:
  SGD(double learningRate = 0.01, double momentum = 0.0) {
    if (momentum < 0 || momentum >= 1)
      throw std::invalid_argument(
          "momentum values must belong to [0, 1) range\n");
    this->learningRate = learningRate;
    this->momentum = momentum;
  }

  SGD(const SGD& obj) {
    this->learningRate = obj.learningRate;
    this->momentum = obj.momentum;
    this->velocity = obj.velocity;
  }

  std::vector<Eigen::MatrixXd> applyOptimzer(
      std::vector<Eigen::MatrixXd> values, std::vector<Eigen::MatrixXd> grads) {
    if (values.size() != grads.size())
      throw std::invalid_argument("values and grads of different size\n");
    bool firstTime = (velocity.size() != grads.size());
    for (int i = 0; i < values.size(); i++) {
      if (firstTime)
        velocity.emplace_back(
            Eigen::MatrixXd::Zero(grads[i].rows(), grads[i].cols()));
      velocity[i] = (momentum * velocity[i]) + ((1 - momentum) * (grads[i]));
      values[i] = values[i] - (learningRate * velocity[i]);
    }
    return values;
  }

  void clear() { velocity.clear(); }
  std::unique_ptr<Optimizer> clone() const {
    return std::make_unique<SGD>(*this);
  }
};
}  // namespace Optimizers

#endif