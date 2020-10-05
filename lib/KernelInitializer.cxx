#include "KernelInitializer.h"

#include <time.h>

#include <iostream>
#include <random>
using namespace std;
using namespace Eigen;

MatrixXd Initializers::HeNormal::generate(int lastLayerSize,
                                          int currentLayerSize) {
  static default_random_engine eng(time(0));
  MatrixXd matrix(currentLayerSize, lastLayerSize + 1);
  matrix << VectorXd::Zero(currentLayerSize),
      MatrixXd::Zero(currentLayerSize, lastLayerSize)
          .unaryExpr([lastLayerSize, currentLayerSize](double dummy) {
            normal_distribution<double> dist(0, sqrt(2.0 / (lastLayerSize)));
            return dist(eng);
          });
  return matrix;
}

unique_ptr<KernelInitializer> Initializers::HeNormal::getKernelInitializer()
    const {
  return make_unique<Initializers::HeNormal>();
}

MatrixXd Initializers::GlorotNormal::generate(int lastLayerSize,
                                              int currentLayerSize) {
  static default_random_engine eng(time(0));
  MatrixXd matrix(currentLayerSize, lastLayerSize + 1);
  matrix << VectorXd::Zero(currentLayerSize),
      MatrixXd::Zero(currentLayerSize, lastLayerSize)
          .unaryExpr([lastLayerSize, currentLayerSize](double dummy) {
            normal_distribution<double> dist(
                0, sqrt(2.0 / (lastLayerSize + currentLayerSize)));
            return dist(eng);
          });
  return matrix;
}

unique_ptr<KernelInitializer> Initializers::GlorotNormal::getKernelInitializer()
    const {
  return make_unique<Initializers::GlorotNormal>();
}