#ifndef ACTIVATION_FUNCTION_H
#define ACTIVATION_FUNCTION_H

#include <bits/stdc++.h>

#include <Eigen/Dense>

class Activation {
 public:
  virtual Eigen::MatrixXd actFn(const Eigen::MatrixXd& matrix) = 0;
  virtual Eigen::MatrixXd actFnGrad(const Eigen::MatrixXd& matrix) = 0;
  virtual std::unique_ptr<Activation> clone() const = 0;
};

namespace Activations {
class Sigmoid : public Activation {
 public:
  Eigen::MatrixXd actFn(const Eigen::MatrixXd& matrix);
  Eigen::MatrixXd actFnGrad(const Eigen::MatrixXd& matrix);
  std::unique_ptr<Activation> clone() const;
};

class Tanh : public Activation {
 public:
  Eigen::MatrixXd actFn(const Eigen::MatrixXd& matrix);
  Eigen::MatrixXd actFnGrad(const Eigen::MatrixXd& matrix);
  std::unique_ptr<Activation> clone() const;
};

class Relu : public Activation {
  const double alpha;

 public:
  Relu(double alpha = 0);
  Eigen::MatrixXd actFn(const Eigen::MatrixXd& matrix);
  Eigen::MatrixXd actFnGrad(const Eigen::MatrixXd& matrix);
  double getAlpha();
  std::unique_ptr<Activation> clone() const;
};

class LeakyRelu : public Relu {
 public:
  LeakyRelu(double alpha = 0.01) : Relu(alpha) {}
};

class Linear : public Relu {
 public:
  Linear() : Relu(1) {}
};
}  // namespace Activations

#endif