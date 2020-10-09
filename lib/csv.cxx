#include "csv.h"

void csv::putData(fstream& fout, vector<vector<double>> data) {
  for (int i = 0; i < data.size(); i++) {
    for (int j = 0; j < data[i].size() - 1; j++) fout << data[i][j] << ",";
    fout << data[i][data[i].size() - 1] << "\n";
  }
  cout << "entered data " << data.size() << " times\n";
}

#ifdef CSV_EIGEN_H
void csv::putData(fstream& fout, MatrixXd data) {
  putData(fout, Conversions::matrixXdToVv(data));
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