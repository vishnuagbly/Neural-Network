#include "csv.h"
using namespace Eigen;
using namespace std;

void csv::putData(fstream& fout, vector<vector<double>> data) {
  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[i].size() - 1; j++) fout << data[i][j] << ",";
    fout << data[i][data[i].size() - 1] << "\n";
  }
  cout << "entered data " << data.size() << " times\n";
}

#ifdef CSV_EIGEN_H
void csv::putData(fstream& fout, MatrixXd data) {
  fout << data.rows() << "," << data.cols() << endl;
  for (int j = 0; j < data.rows(); j++)
    for (int k = 0; k < data.cols(); k++) {
      if (j == data.rows() - 1 && k == data.cols() - 1)
        fout << data(j, k) << endl;
      else
        fout << data(j, k) << ",";
    }
}

// returns matrix of size 0 if no line read.
MatrixXd csv::getMatrixXd(fstream& fin) {
  string temp;
  getline(fin, temp);
  if (temp == "") return MatrixXd();
  auto dim = getValuesFromCsvLine(temp);
  MatrixXd matrix((int)dim[0], (int)dim[1]);
  temp.clear();
  getline(fin, temp);
  auto values = getValuesFromCsvLine(temp);
  int k = 0;
  for (int i = 0; i < matrix.rows(); i++)
    for (int j = 0; j < matrix.cols(); j++) matrix(i, j) = values[k++];
  return matrix;
}
#endif

vector<vector<double>> csv::getData(fstream& fin) {
  string temp;
  vector<vector<double>> res;
  while (fin >> temp) {
    string line;
    string word;
    vector<double> currentInputs;
    getline(fin, line);
    stringstream s(line);
    while (getline(s, word, ',')) currentInputs.emplace_back(stod(word));
    res.emplace_back(currentInputs);
  }
  return res;
}

vector<double> csv::getValuesFromCsvLine(string temp) {
  stringstream tempStream(temp);
  string tempWord;
  vector<double> values;
  while (getline(tempStream, tempWord, ',')) {
    values.emplace_back(stod(tempWord));
  }
  return values;
}