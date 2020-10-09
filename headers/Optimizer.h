#ifndef OPTIMIZER_H
#define OPTIMIZER_H

template <typename T>
class Optimizer {
 public:
  virtual T applyOptimzer(T values, T grads) = 0;
  virtual std::unique_ptr<Optimizer<T>> clone() const = 0;
};

namespace Optimizers {
/// SGD assumes T type can perform basic arithemetic operations.
template <typename T>
class SGD : public Optimizer<T> {
  double learningRate;

 public:
  SGD(double learningRate = 0.01) { this->learningRate = learningRate; }
  T applyOptimzer(T values, T grads) { return values - (learningRate * grads); }
  std::unique_ptr<Optimizer<T>> clone() const {
    return std::make_unique<SGD<T>>(learningRate);
  }
};
}  // namespace Optimizers

#endif