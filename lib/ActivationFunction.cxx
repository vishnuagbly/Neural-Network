#include "ActivationFunction.h"

#define DBL_EPSILON_RANGE (1 - (2 * DBL_EPSILON))

using namespace Activations;
using namespace Eigen;
using namespace std;

MatrixXd Sigmoid::actFn(const MatrixXd& matrix) {
  MatrixXd res(matrix.rows(), matrix.cols());
  MatrixXd ones = MatrixXd::Ones(matrix.rows(), matrix.cols());
  res = ones + (-matrix).array().exp().matrix();
  res = ones.cwiseQuotient(res);
  res = DBL_EPSILON_RANGE * res + DBL_EPSILON * ones;
  return res;
}

MatrixXd Sigmoid::actFnGrad(const MatrixXd& matrix) {
  return (1 - 1e-16 - 1e-300) *
         matrix.cwiseProduct(MatrixXd::Ones(matrix.rows(), matrix.cols()) -
                             matrix);
}

unique_ptr<Activation> Sigmoid::clone() const { return make_unique<Sigmoid>(); }

MatrixXd Tanh::actFn(const MatrixXd& matrix) {
  MatrixXd res = MatrixXd::Zero(matrix.rows(), matrix.cols());
  for (int i = 0; i < res.rows(); i++) {
    for (int j = 0; j < res.cols(); j++) {
      double value = matrix(i, j);
      double eZ = exp(value), eNZ = exp(-value);
      double temp;
      if (eZ == INFINITY)
        temp = 1;
      else if (eNZ == INFINITY)
        temp = -1;
      else
        temp = (eZ - eNZ) / (eZ + eNZ);
      temp *= DBL_EPSILON_RANGE;
      if (eZ >= eNZ)
        temp += DBL_EPSILON;
      else
        temp -= DBL_EPSILON;
      res(i, j) = temp;
    }
  }
  return res;
}

MatrixXd Tanh::actFnGrad(const MatrixXd& matrix) {
  auto res = actFn(matrix);
  return DBL_EPSILON_RANGE *
         (MatrixXd::Ones(matrix.rows(), matrix.cols()) - res.cwiseProduct(res));
}

unique_ptr<Activation> Tanh::clone() const { return make_unique<Tanh>(); }

Relu::Relu(double alpha) : alpha(alpha) {}

MatrixXd Relu::actFn(const MatrixXd& matrix) {
  MatrixXd res = (matrix + matrix.cwiseAbs()) / 2;
  res += alpha * ((matrix - matrix.cwiseAbs()) / 2);
  return res;
}

MatrixXd Relu::actFnGrad(const MatrixXd& matrix) {
  MatrixXd res(matrix.rows(), matrix.cols());
  for (int i = 0; i < matrix.rows(); i++)
    for (int j = 0; j < matrix.cols(); j++)
      res(i, j) = (matrix(i, j) > 0) + (alpha * (matrix(i, j) <= 0));
  return res;
}

double Relu::getAlpha() { return alpha; }

unique_ptr<Activation> Relu::clone() const { return make_unique<Relu>(alpha); }