#ifndef CSV_H
#define CSV_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>
using namespace std;

namespace csv {
void putData(fstream& fout, vector<vector<double>> data);
vector<vector<double>> getData(fstream& fin);
}  // namespace csv

#ifdef __has_include
#if __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#ifdef __has_include
#if __has_include("EigenStlConversions.h")
#include "EigenStlConversions.h"
#ifndef CSV_EIGEN_H
#define CSV_EIGEN_H
namespace csv {
void putData(fstream& fout, Eigen::MatrixXd data);
}  // namespace csv
#endif  // CSV_EIGEN_H
#endif
#endif  // ifdef EigenStlConversions
#endif
#endif  // ifdef Eigen/Dense

#endif