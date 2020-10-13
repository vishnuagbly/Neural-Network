#ifndef COST_FUNCTION_H
#define COST_FUNCTION_H
#include <Eigen/Dense>
#include <memory>

class CostFunction {
 public:
  virtual double costFn(const Eigen::VectorXd& output,
                        const Eigen::VectorXd& expected) const = 0;

  virtual Eigen::VectorXd costFnGrad(const Eigen::VectorXd& output,
                                     const Eigen::VectorXd& expected) const = 0;

  virtual std::unique_ptr<CostFunction> clone() const = 0;
};

namespace CostFns {
class CrossEntropy : public CostFunction {
  double calcSingleCost(double output, double expected) const;

 public:
  double costFn(const Eigen::VectorXd& output,
                const Eigen::VectorXd& expected) const;

  Eigen::VectorXd costFnGrad(const Eigen::VectorXd& output,
                             const Eigen::VectorXd& expected) const;

  std::unique_ptr<CostFunction> clone() const;
};

class Quadratic : public CostFunction {
 public:
  double costFn(const Eigen::VectorXd& output,
                const Eigen::VectorXd& expected) const;

  Eigen::VectorXd costFnGrad(const Eigen::VectorXd& output,
                             const Eigen::VectorXd& expected) const;

  std::unique_ptr<CostFunction> clone() const;
};
}  // namespace CostFns

#endif