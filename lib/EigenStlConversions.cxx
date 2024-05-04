#include "EigenStlConversions.h"

MatrixXd Conversions::vVToMatrixXd(vector<vector<double>> data) {
  if (!data.size()) return MatrixXd();
  MatrixXd res(data.size(), data[0].size());
  for (int i = 0; i < data.size(); i++) {
    if (data[i].size() != data[0].size())
      throw invalid_argument("wrong row size of vector");
    for (int j = 0; j < data[0].size(); j++) res(i, j) = data[i][j];
  }
  return res;
}

vector<vector<double>> Conversions::matrixXdToVv(MatrixXd data) {
  vector<vector<double>> res;
  for (int i = 0; i < data.rows(); i++) {
    vector<double> temp;
    for (int j = 0; j < data.cols(); j++) temp.emplace_back(data(i, j));
    res.emplace_back(temp);
  }
  return res;
}