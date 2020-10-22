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
  SGD(double learningRate = 0.01, double momentum = 0.0);

  SGD(const SGD& obj);

  std::vector<Eigen::MatrixXd> applyOptimzer(
      std::vector<Eigen::MatrixXd> values, std::vector<Eigen::MatrixXd> grads);

  void clear();
  std::unique_ptr<Optimizer> clone() const;
};

class Adam : public Optimizer {
  double alpha;
  double beta1;
  double beta2;
  double epsilon = 1e-8;
  std::vector<Eigen::MatrixXd> moment1, moment2;

 public:
  Adam(double alpha = 0.001, double beta1 = 0.9, double beta2 = 0.999);

  Adam(const Adam& obj);

  std::vector<Eigen::MatrixXd> applyOptimzer(
      std::vector<Eigen::MatrixXd> values, std::vector<Eigen::MatrixXd> grads);

  void clear();
  std::unique_ptr<Optimizer> clone() const;
};
}  // namespace Optimizers

#endif