#ifndef CSV_H
#define CSV_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <vector>

namespace csv {
void putData(std::fstream& fout, std::vector<std::vector<double>> data);
std::vector<std::vector<double>> getData(std::fstream& fin);
std::vector<double> getValuesFromCsvLine(std::string temp);
}  // namespace csv

#ifdef __has_include
#if __has_include(<Eigen/Dense>)
#include <Eigen/Dense>
#ifndef CSV_EIGEN_H
#define CSV_EIGEN_H
namespace csv {
void putData(std::fstream& fout, Eigen::MatrixXd data);
Eigen::MatrixXd getMatrixXd(std::fstream& fin);
}  // namespace csv
#endif  // CSV_EIGEN_H
#endif
#endif  // ifdef Eigen/Dense

#endif