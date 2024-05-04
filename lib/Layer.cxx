#include "Layer.h"

using namespace Eigen;
using namespace std;

Layer::Layer(int units) { this->units = units; }

int Layer::size() const { return units; }

MatrixXd Layer::getBias(const MatrixXd& matrix) {
  return matrix.block(0, 0, matrix.rows(), 1);
}

MatrixXd Layer::getKernel(const MatrixXd& matrix) {
  return matrix.block(0, 1, matrix.rows(), matrix.cols() - 1);
}