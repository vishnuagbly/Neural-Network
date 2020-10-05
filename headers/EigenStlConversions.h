#ifndef EIGENSTLCONVERSIONS_H
#define EIGENSTLCONVERSIONS_H
#include <bits/stdc++.h>

#include <Eigen/Dense>
using namespace Eigen;
using namespace std;

namespace Conversions {
MatrixXd vVToMatrixXd(vector<vector<double>> data);
vector<vector<double>> matrixXdToVv(MatrixXd data);
}  // namespace Conversions
#endif