#include "ActivationFunction.h"
using namespace Activations;
using namespace Eigen;
using namespace std;

MatrixXd Sigmoid::actFn(const MatrixXd& matrix) {
  MatrixXd res(matrix.rows(), matrix.cols());
  MatrixXd ones = MatrixXd::Ones(matrix.rows(), matrix.cols());
  res << ones + (-matrix).array().exp().matrix();
  res << ones.cwiseQuotient(res);
  return res;
}

MatrixXd Sigmoid::actFnGrad(const MatrixXd& matrix) {
  return matrix.cwiseProduct(MatrixXd::Ones(matrix.rows(), matrix.cols()) -
                             matrix);
}

unique_ptr<Activation> Sigmoid::clone() const { return make_unique<Sigmoid>(); }

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