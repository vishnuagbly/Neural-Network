#include "CostFunction.h"

#include <iostream>
using namespace std;
using namespace Eigen;

double CostFns::CrossEntropy::costFn(const VectorXd& output,
                                     const VectorXd& expected) const {
  double costSum = 0;
  for (int k = 0; k < output.size(); k++) {
    double temp = costSum;
    costSum += calcSingleCost(output[k], expected[k]);
    if (costSum != costSum) {
      cout << "costSum after input row[" << k << "]: " << costSum << endl;
      cout << "returned value: " << calcSingleCost(output[k], expected[k])
           << endl;
      cout << "temp: " << temp << endl;
      cout << "expected:\n" << expected << endl;
      cout << "output:\n" << output << endl;
      return costSum;
    }
  }
  return -costSum;
}

double CostFns::CrossEntropy::calcSingleCost(double output,
                                             double expected) const {
  if (expected)
    return log(output);
  else
    return log(1 - output);
}

VectorXd CostFns::CrossEntropy::calcOutputErr(
    const VectorXd& output, const VectorXd& expected,
    const VectorXd& actFnGradOutput) const {
  auto denom = output.cwiseProduct(VectorXd::Ones(output.size()) - output);
  auto neum = output - expected;

  if (denom.isApprox(actFnGradOutput)) return neum;

  auto grad = neum.cwiseQuotient(denom);
  return grad.cwiseProduct(actFnGradOutput);
}

unique_ptr<CostFunction> CostFns::CrossEntropy::clone() const {
  return make_unique<CostFns::CrossEntropy>();
}

double CostFns::Quadratic::costFn(const VectorXd& output,
                                  const VectorXd& expected) const {
  VectorXd vect = output - expected;
  vect = vect.cwiseProduct(vect) * 0.5;
  if (vect.any() < 0) throw runtime_error("negative value not possible\n");
  return vect.sum();
}

VectorXd CostFns::Quadratic::calcOutputErr(
    const VectorXd& output, const VectorXd& expected,
    const VectorXd& actFnGradOutput) const {
  auto res = (output - expected).cwiseProduct(actFnGradOutput);
  if (res.any() > 1000) {
    cout << "output: " << output.transpose() << endl;
    cout << "expected: " << expected.transpose() << endl;
    cout << "actFnGradOutput: " << expected.transpose() << endl;
    cout << "res: " << res.transpose() << endl;
    throw runtime_error("too large error");
  }
  return res;
}

unique_ptr<CostFunction> CostFns::Quadratic::clone() const {
  return make_unique<CostFns::Quadratic>();
}
