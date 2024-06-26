#include "NeuralNetwork.h"

using namespace Eigen;
using namespace std;

namespace detail {
template <typename T>  // is_float         is_unsigned
inline std::string to_string(T const val, std::false_type, std::false_type) {
  return std::to_string(static_cast<long long>(val));
}

template <typename T>  // is_float         is_unsigned
inline std::string to_string(T const val, std::false_type, std::true_type) {
  return std::to_string(static_cast<unsigned long long>(val));
}

template <typename T, typename _>  // is_float
inline std::string to_string(T const val, std::true_type, _) {
  return std::to_string(static_cast<long double>(val));
}
}  // namespace detail

template <typename T>
inline std::string to_string(T const val) {
  return detail::to_string(val, std::is_floating_point<T>(),
                           std::is_unsigned<T>());
}

VectorXd operator-(const double scalar, const VectorXd& vect) {
  return vect - (scalar * VectorXd::Ones(vect.size()));
}

vector<MatrixXd> operator+(const vector<MatrixXd>& first,
                           const vector<MatrixXd>& second) {
  if (first.size() != second.size())
    throw invalid_argument(
        "first and second does not contain eq amount of matrices\n");

  vector<MatrixXd> res;
  for (int i = 0; i < first.size(); i++) res.emplace_back(first[i] + second[i]);
  return res;
}

void print(vector<double> arr) {
  for (double value : arr) cout << value << " ";
  cout << endl;
}

bool checkForNan(MatrixXd matrix) {
  for (int i = 0; i < matrix.rows(); i++)
    for (int j = 0; j < matrix.cols(); j++)
      if (matrix(i, j) != matrix(i, j)) return true;
  return false;
}

bool checkForNan(vector<MatrixXd> value) {
  for (int i = 0; i < value.size(); i++)
    if (checkForNan(value[i])) return true;
  return false;
}

bool checkForNan(vector<VectorXd> value) {
  for (int i = 0; i < value.size(); i++)
    if (checkForNan(value[i])) return true;
  return false;
}

NeuralNetwork::NeuralNetwork(const vector<int>& layerSizes,
                             vector<MatrixXd> weights, int expectedOutputSize,
                             const CostFunction& costFn,
                             const Activation& activation,
                             const Optimizer& optimizer) {
  if (layerSizes.size() < 2)
    throw invalid_argument("there should be atleast 2 layers");
  if (any_of(layerSizes.begin(), layerSizes.end(),
             [](int size) { return size < 1; }))
    throw invalid_argument("each layer size should atleast be 1");
  if (weights.size() != layerSizes.size() - 1)
    throw invalid_argument(
        "length of weight matrices must be eq. to total layers - 1");
  for (int i = 0; i < layerSizes.size() - 1; i++) {
    if (checkForNan(weights[i])) throw invalid_argument("nan found in weights");
    if (weights[i].rows() != layerSizes[i + 1])
      throw invalid_argument(
          "weight matrices does not have correct number of rows");
    if (weights[i].cols() != layerSizes[i] + 1)
      throw invalid_argument(
          "weight matrices does not have correct number of columns");
  }
  if (expectedOutputSize != -1 &&
      dynamic_cast<const CostFns::CrossEntropy*>(&costFn) != nullptr)
    throw invalid_argument(
        "please define both expectedOutputSize and outputErr fn together");

  for (int i = 0; i < layerSizes.size(); i++) {
    if (i > 0 && i < layerSizes.size() - 1)
      this->layersProps.emplace_back(make_unique<Layers::Dense>(
          layerSizes[i], layerSizes[i - 1], weights[i - 1], activation));
    else if (i == layerSizes.size() - 1) {
      this->layersProps.emplace_back(
          make_unique<Layers::Dense>(layerSizes[i], layerSizes[i - 1],
                                     weights[i - 1], Activations::Sigmoid()));
    } else
      this->layersProps.emplace_back(make_unique<Layers::Input>(layerSizes[i]));
  }
  if (expectedOutputSize == -1) this->expectedOutputSize = layerSizes.back();
  this->costFn = costFn.clone();
  this->optimizer = optimizer.clone();
}

NeuralNetwork::NeuralNetwork(vector<unique_ptr<Layer>>& layers,
                             const Optimizer& optimizer,
                             const CostFunction& costFn) {
  if (layers.size() < 2)
    throw invalid_argument("there should atleast be 2 layers");
  if (layers[0]->getName() != Layers::Input::name)
    throw invalid_argument("first layer should be of inputType not " +
                           layers[0]->getName());
  for (int i = 0; i < layers.size(); i++) {
    this->layersProps.emplace_back(std::move(layers[i]));
    if (i > 0) {
      if (layersProps[i]->getName() == Layers::Input::name)
        throw invalid_argument("only first layer can be input layer\n");
      layersProps[i]->initializeWeights(layersProps[i - 1]->size());
    }
  }
  this->optimizer = optimizer.clone();
  this->expectedOutputSize = layersProps.back()->size();
  this->costFn = costFn.clone();
}

NeuralNetwork::NeuralNetwork(vector<unique_ptr<Layer>>& layers, fstream& fin,
                             const Optimizer& optimizer,
                             const CostFunction& costFn)
    : NeuralNetwork(layers, optimizer, costFn) {
  uploadWeights(fin);
}

NeuralNetwork::NeuralNetwork(const CostFunction& costFn,
                             const Optimizer& optimizer) {
  this->costFn = costFn.clone();
  this->optimizer = optimizer.clone();
}

NeuralNetwork::NeuralNetwork(const NeuralNetwork& nn) {
  for (int i = 0; i < nn.layersProps.size(); i++) {
    this->layersProps.emplace_back(nn.layersProps[i]->clone());
  }
  this->expectedOutputSize = nn.expectedOutputSize;
  this->costFn = nn.costFn->clone();
  this->optimizer = nn.optimizer->clone();
}

void NeuralNetwork::add(const Layer& layer) {
  if (!layersProps.size() && layer.getName() != Layers::Input::name)
    throw invalid_argument("first layer should be an input layer\n");
  else if (layersProps.size() >= 1 && layer.getName() == Layers::Input::name)
    throw invalid_argument("only first layer can be an input layer\n");
  layersProps.emplace_back(layer.clone());
  if (layersProps.size() > 1) {
    layersProps.back()->initializeWeights(
        layersProps[layersProps.size() - 2]->size());
    this->expectedOutputSize = layersProps.back()->size();
  }
}

void NeuralNetwork::uploadWeights(fstream& fin) {
  auto weights = getWeights(fin);
  if (weights.size() != layersProps.size() - 1)
    throw invalid_argument("weights not of correct size\n");
  for (int i = 1; i < totalLayers(); i++) {
    layersProps[i]->updateWeights(weights[i - 1]);
  }
}

MatrixXd NeuralNetwork::printResults(
    const MatrixXd& inputData, const MatrixXd& expectedOutput,
    const double lambda,
    double (*accuracyFn)(const VectorXd& outputData,
                         const VectorXd& expectedOutput)) const {
  assertInputAndOutputData(inputData, expectedOutput);
  MatrixXd outputData(expectedOutput.rows(), expectedOutput.cols());
  for (int i = 0; i < expectedOutput.rows(); i++) {
    outputData.row(i) << getOutput(inputData.row(i));
  }
  auto cost = calcTotalCost(outputData, expectedOutput, lambda);
  cout << "total cost: " << cost << endl;
  auto accuracy =
      getTotalAccuracy(inputData, outputData, expectedOutput, accuracyFn);
  // cout << "outputData:\n" << outputData << "\n\n";
  cout << "accuracy: " << accuracy << "\n\n";
  return outputData;
}

MatrixXd NeuralNetwork::allOutputs(const MatrixXd& inputData) const {
  MatrixXd outputData(inputData.rows(), layersProps.back()->size());
  for (int i = 0; i < inputData.rows(); i++) {
    outputData.row(i) = getOutput(inputData.row(i));
  }
  return outputData;
}

double NeuralNetwork::calcTotalCost(const MatrixXd& outputData,
                                    const MatrixXd& expectedOutputData,
                                    const double lambda) const {
  assertOutputData(expectedOutputData);
  double costSum = 0;
  for (int i = 0; i < outputData.rows(); i++) {
    VectorXd lastLayerValues = outputData.row(i);
    // cout << "input[" << i << "]: lasy layer values\n"
    //      << lastLayerValues << endl;
    costSum += costFn->costFn(lastLayerValues, expectedOutputData.row(i));
  }
  // cout << endl;
  double regularisationTerm = 0;
  for (int i = 1; i < totalLayers(); i++) {
    MatrixXd tempWeights = layersProps[i]->getWeights();
    double temp = regularisationTerm;
    regularisationTerm += tempWeights.cwiseProduct(tempWeights).sum();
  }
  if (regularisationTerm == INFINITY) regularisationTerm = DBL_MAX;
  costSum += (lambda / 2) * regularisationTerm;
  costSum /= outputData.rows();
  return costSum;
}

MatrixXd NeuralNetwork::getOutput(const MatrixXd& inputData) const {
  MatrixXd output(inputData.rows(), layersProps.back()->size());
  for (int i = 0; i < inputData.rows(); i++)
    output.row(i) = calcAllNodes(inputData.row(i)).back();
  return output;
}

double NeuralNetwork::getTotalAccuracy(
    const MatrixXd& inputData, const MatrixXd& outputData,
    const MatrixXd& expectedOutput,
    double (*accuracyFn)(const VectorXd& outputData,
                         const VectorXd& expectedOutput)) {
  double totalCorrect = 0;
  for (int i = 0; i < inputData.rows(); i++) {
    totalCorrect += accuracyFn(outputData.row(i), expectedOutput.row(i));
  }
  return (totalCorrect / inputData.rows()) * 100;
}

double NeuralNetwork::getAccuracy(const VectorXd& outputData,
                                  const VectorXd& expectedOutput) {
  VectorXd::Index i1, i2;
  outputData.maxCoeff(&i1);
  expectedOutput.maxCoeff(&i2);
  return i1 == i2;
}

vector<vector<double>> NeuralNetwork::train(
    const MatrixXd& inputData, const MatrixXd& outputData, const double lambda,
    const int batchSize, const int totalRounds,
    const bool printWeightsAndLastChange, int costRecordInterval) {
  assertLambda(lambda);
  assertInputAndOutputData(inputData, outputData);
  if (costRecordInterval <= 0)
    throw invalid_argument(
        "costRecordInterval cannot be equal to or less than zero");
  vector<MatrixXd> decBy;
  vector<vector<double>> res;
  int k = 0;
  int itr = 0;
  optimizer->clone();
  calcTotalThreadsToUse(batchSize);
  for (int round = 0; round < totalRounds; round++) {
    // cout << "weights:\n";
    // printWeights();
    // cout << "------------------------------\n";
    for (int i = 0; i < inputData.rows(); i++) {
      decBy.clear();
      for (int j = 0; j < totalLayers() - 1; j++)
        decBy.emplace_back(
            MatrixXd::Zero(layersProps[j + 1]->getWeights().rows(),
                           layersProps[j + 1]->getWeights().cols()));

      vector<future<vector<MatrixXd>>> threads(totalThreads - 1);

      auto isolateTrainingPtr = &NeuralNetwork::isolateTraining;
      int batchElement = 0;
      for (int th = 0; th < threads.size(); th++) {
        threads[th] =
            async(isolateTrainingPtr, *this, ref(inputData), ref(outputData),
                  lambda, round, batchSize, th, threads.size() + 1, i);
      }
      decBy = isolateTraining(inputData, outputData, lambda, round, batchSize,
                              threads.size(), threads.size() + 1, i);
      for (int th = 0; th < threads.size(); th++)
        decBy = decBy + threads[th].get();

      int incrementBy = min(batchSize, (int)inputData.rows() - i);
      batchElement += incrementBy;
      i += incrementBy;

      // for (; batchElement < batchSize && i < inputData.rows();
      //      batchElement++, i++) {
      //   try {
      //     decBy = getDecBy(inputData.row(i), outputData.row(i), lambda,
      //     decBy);
      //   } catch (exception& err) {
      //     cout << "batchElement: " << batchElement << endl;
      //     cout << "input row i: " << i << endl;
      //     cout << "round: " << round << endl;
      //     throw err;
      //   }
      // }
      decBy = updateWeights(decBy, batchElement);
      if (!(itr++ % costRecordInterval) ||
          itr == inputData.rows() * totalRounds - 1) {
        vector<double> temp = {(double)itr, calcTotalCost(allOutputs(inputData),
                                                          outputData, lambda)};
        res.emplace_back(temp);
      }
    }
  }
  if (printWeightsAndLastChange) printWeightsAndUpdates(decBy);
  return res;
}

void NeuralNetwork::calcTotalThreadsToUse(const int batchSize) {
  int maxThreads = thread::hardware_concurrency();
  if (totalThreads <= maxThreads && totalThreads > 0) return;
  if (batchSize < 20) {
    totalThreads = 1;
    return;
  }
  totalThreads = batchSize / 20;
  totalThreads = max(min(maxThreads, totalThreads), 1);
}

vector<MatrixXd> NeuralNetwork::isolateTraining(const MatrixXd& inputData,
                                                const MatrixXd& outputData,
                                                int lambda, int round,
                                                int batchSize, int index,
                                                int totalThreads,
                                                int currentRow) const {
  vector<MatrixXd> decBy;
  for (int j = 0; j < totalLayers() - 1; j++)
    decBy.emplace_back(MatrixXd::Zero(layersProps[j + 1]->getWeights().rows(),
                                      layersProps[j + 1]->getWeights().cols()));

  int i = currentRow + index;
  for (int batchElement = index;
       batchElement < batchSize && i < inputData.rows();
       i += totalThreads, batchElement += totalThreads) {
    try {
      decBy = getDecBy(inputData.row(i), outputData.row(i), lambda, decBy);
    } catch (exception& err) {
      cout << "batchElement: " << batchElement << endl;
      cout << "input row i: " << i << endl;
      cout << "round: " << round << endl;
      throw err;
    }
  }
  return decBy;
}

vector<MatrixXd> NeuralNetwork::getDecBy(const VectorXd& inputData,
                                         const VectorXd& outputData,
                                         const double lambda,
                                         vector<MatrixXd> decBy) const {
  vector<VectorXd> layersValues;
  try {
    layersValues = calcAllNodes(inputData);
    if (checkForNan(layersValues)) throw runtime_error("layerValues nan found");
  } catch (exception& err) {
    cout << "found err in calc All Nodes\n";
    throw err;
  }
  vector<VectorXd> dell;
  try {
    dell = calcDell(outputData, layersValues);
    if (checkForNan(dell)) throw runtime_error("dell nan found");
  } catch (exception& err) {
    cout << "found err in calc dell\n";
    throw err;
  }
  try {
    decBy = calcBigDell(dell, layersValues, lambda, decBy);
    if (checkForNan(decBy)) throw runtime_error("decBy nan found");
  } catch (exception& err) {
    cout << "found err in calcBigDell\n";
    throw err;
  }
  return decBy;
}

vector<VectorXd> NeuralNetwork::calcAllNodes(const VectorXd& inputData) const {
  if (inputData.size() != layersProps.front()->size())
    throw invalid_argument("Input of wrong size");

  vector<VectorXd> layersValues;
  layersValues.emplace_back(inputData);
  for (int j = 1; j < totalLayers(); j++) {
    VectorXd layerValues = layersProps[j]->getOutputValues(layersValues.back());

    if (checkForNan(layerValues)) {
      cout << "lastLayer:\n" << layersValues.back() << endl;
      cout << "zValues[" << j << "]:\n"
           << layersProps[j]->getZValues(layersValues.back()) << endl;
      cout << "weights[" << j << "]:\n" << layersProps[j]->getWeights() << endl;
      cout << "layer[" << j << "]:\n" << layerValues << endl;
      cout << "actFn: "
           << layersProps[j]->getActivation()->actFn(
                  layersProps[j]->getZValues(layersValues.back()))
           << endl;
      throw runtime_error("nan found");
    }
    layersValues.emplace_back(layerValues);
  }
  // printLayersValues(layersValues);
  return layersValues;
}

vector<VectorXd> NeuralNetwork::calcDell(
    const VectorXd& outputData, const vector<VectorXd>& layersValues) const {
  if (outputData.size() != layersProps.back()->size())
    throw invalid_argument("wrong outputData size");
  vector<VectorXd> dell;
  auto costGrad = costFn->costFnGrad(layersValues.back(), outputData);
  auto lastLayerGrad =
      layersProps.back()->getActivation()->actFnGrad(layersValues.back());
  dell.emplace_back(costGrad.cwiseProduct(lastLayerGrad));

  // if (costGrad.cwiseAbs().maxCoeff() > 150) {
  //   cout << "output: " << layersValues.back().transpose() << endl;
  //   cout << "expected: " << outputData.transpose() << endl;
  //   cout << "dell last layer: " << dell.back().transpose() << endl;
  //   throw runtime_error("got dell > 200\n");
  // }

  for (int j = totalLayers() - 2; j > 0; j--) {
    VectorXd temp = layersProps[j + 1]->backPropagateDell(dell.front());
    VectorXd dellLayer = temp.cwiseProduct(
        layersProps[j]->getActivation()->actFnGrad(layersValues[j]));
    dell.insert(dell.begin(), dellLayer);
  }
  dell.insert(dell.begin(), VectorXd::Zero(layersProps[0]->size()));
  return dell;
}

vector<MatrixXd> NeuralNetwork::calcBigDell(
    const vector<VectorXd>& dell, const vector<VectorXd>& layersValues,
    const double lambda, vector<MatrixXd> decBy) const {
  assertLambda(lambda);
  for (int i = 0; i < totalLayers() - 1; i++) {
    decBy[i] +=
        layersProps[i + 1]->calcBigDell(dell[i + 1], layersValues[i], lambda);
  }
  return decBy;
}

void NeuralNetwork::printWeightsAndUpdates(vector<MatrixXd> decBy) const {
  for (int i = 1; i < totalLayers(); i++) {
    cout << "weights[" << i << "]:\n";
    cout << layersProps[i]->getWeights() << "\n\n";
    cout << "updates[" << i - 1 << "]:\n";
    cout << decBy[i - 1] << "\n\n";
  }
}

void NeuralNetwork::printWeights() const {
  for (int i = 1; i < totalLayers(); i++) {
    cout << "weights[" << i << "]:\n";
    cout << layersProps[i]->getWeights() << "\n\n";
  }
}

void NeuralNetwork::putWeights(string filePath) {
  fstream fout(filePath, fstream::out | fstream::trunc);
  fout.seekp(0);
  fout.seekg(0);
  auto weights = getWeights();
  for (int i = 0; i < weights.size(); i++) {
    csv::putData(fout, weights[i]);
  }
}

vector<MatrixXd> NeuralNetwork::getWeights() {
  vector<MatrixXd> res;
  for (int i = 1; i < totalLayers(); i++)
    res.emplace_back(layersProps[i]->getWeights());
  return res;
}

vector<MatrixXd> NeuralNetwork::getWeights(fstream& fin) {
  vector<MatrixXd> res;
  string check;
  while (fin) {
    MatrixXd matrix = csv::getMatrixXd(fin);
    if (!matrix.size()) continue;
    res.emplace_back(matrix);
  }
  return res;
}

vector<MatrixXd> NeuralNetwork::updateWeights(vector<MatrixXd> decBy,
                                              const double totalInputs) {
  vector<MatrixXd> weights;
  for (int i = 1; i < totalLayers(); i++) {
    decBy[i - 1] = decBy[i - 1] / totalInputs;
    weights.emplace_back(layersProps[i]->getWeights());
  }
  weights = optimizer->applyOptimzer(weights, decBy);
  for (int i = 1; i < totalLayers(); i++)
    layersProps[i]->updateWeights(weights[i - 1]);
  return decBy;
}

unique_ptr<NeuralNetwork> NeuralNetwork::clone() const {
  return make_unique<NeuralNetwork>(*this);
}

void NeuralNetwork::assertInputAndOutputData(const MatrixXd& inputData,
                                             const MatrixXd& outputData) const {
  if (inputData.rows() != outputData.rows())
    throw invalid_argument("inputData and outputData not of equal sizes");
  assertInputData(inputData);
  assertOutputData(outputData);
}

void NeuralNetwork::assertInputData(const MatrixXd& inputData) const {
  if (inputData.cols() != layersProps[0]->size())
    throw invalid_argument("each input should be of size " +
                           to_string(layersProps[0]->size()) + " not of size " +
                           to_string(inputData.cols()));
}

void NeuralNetwork::assertOutputData(const MatrixXd& outputData) const {
  if (outputData.cols() != expectedOutputSize)
    throw invalid_argument("each output should be of size " +
                           to_string(layersProps.back()->size()) + "and not " +
                           to_string(outputData.cols()));
}

void NeuralNetwork::assertLambda(const double lambda) const {
  if (lambda < 0) throw invalid_argument("lambda cannot be less than zero");
}

void printLayersValues(vector<VectorXd> layersValues) {
  cout << "layers:\n";
  int maxLayerSize = layersValues[0].size();
  for (int i = 0; i < layersValues.size(); i++)
    if (maxLayerSize < layersValues[i].size())
      maxLayerSize = layersValues[i].size();
  cout << "maxLayerSize: " << maxLayerSize << endl;
  MatrixXd layers(maxLayerSize, layersValues.size());
  for (int i = 0; i < maxLayerSize; i++) {
    for (int j = 0; j < layersValues.size(); j++) {
      if (i >= layersValues[j].size())
        layers(i, j) = -1;
      else
        layers(i, j) = layersValues[j][i];
    }
  }
  cout << layers << "\n\n";
}
