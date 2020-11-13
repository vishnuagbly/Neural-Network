#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#pragma once
#include <Eigen/Dense>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>

#include "CostFunction.h"
#include "Layer.h"
#include "Optimizer.h"
#include "csv.h"
void printLayersValues(std::vector<Eigen::VectorXd> layersValues);

class NeuralNetwork {
  std::vector<std::unique_ptr<Layer>> layersProps;
  int expectedOutputSize;
  //   Eigen::VectorXd (*calcOutputErr)(const Eigen::VectorXd& outputData,
  //                                    const Eigen::VectorXd& lastLayer);
  std::unique_ptr<CostFunction> costFn;
  std::unique_ptr<Optimizer> optimizer;

  int totalLayers() const { return layersProps.size(); }

  std::vector<Eigen::VectorXd> calcAllNodes(
      const Eigen::VectorXd& inputData) const;

  std::vector<Eigen::VectorXd> calcDell(
      const Eigen::VectorXd& outputData,
      const std::vector<Eigen::VectorXd>& layersValues) const;

  std::vector<Eigen::MatrixXd> calcBigDell(
      const std::vector<Eigen::VectorXd>& dell,
      const std::vector<Eigen::VectorXd>& layersValues, const double lambda,
      std::vector<Eigen::MatrixXd> decBy) const;

  std::vector<Eigen::MatrixXd> updateWeights(std::vector<Eigen::MatrixXd> decBy,
                                             const double totalInputs);

 public:
  NeuralNetwork(const std::vector<int>& layerSizes,
                std::vector<Eigen::MatrixXd> weights,
                int expectedOutputSize = -1,
                const CostFunction& costFn = CostFns::CrossEntropy(),
                const Activation& activation = Activations::Sigmoid(),
                const Optimizer& optimizer = Optimizers::SGD());

  NeuralNetwork(const std::vector<int>& layerSizes,
                std::vector<Eigen::MatrixXd> weights,
                const Activation& activation,
                const Optimizer& optimizer = Optimizers::SGD())
      : NeuralNetwork(layerSizes, weights, -1, CostFns::CrossEntropy(),
                      activation, optimizer) {}

  NeuralNetwork(const std::vector<int>& layerSizes,
                std::vector<Eigen::MatrixXd> weights,
                const Optimizer& optimizer = Optimizers::SGD(),
                const Activation& activation = Activations::Sigmoid())
      : NeuralNetwork(layerSizes, weights, activation, optimizer) {}

  NeuralNetwork(std::vector<std::unique_ptr<Layer>>& layers,
                const Optimizer& optimizer = Optimizers::SGD(),
                const CostFunction& costFn = CostFns::CrossEntropy());

  NeuralNetwork(std::vector<std::unique_ptr<Layer>>& layers, std::fstream& fin,
                const Optimizer& optimizer = Optimizers::SGD(),
                const CostFunction& costFn = CostFns::CrossEntropy());

  NeuralNetwork(std::vector<std::unique_ptr<Layer>>& layers,
                const CostFunction& costFn,
                const Optimizer& optimizer = Optimizers::SGD())
      : NeuralNetwork(layers, optimizer, costFn) {}

  NeuralNetwork(const CostFunction& costFn = CostFns::CrossEntropy(),
                const Optimizer& optimizer = Optimizers::SGD());

  NeuralNetwork(const Optimizer& optmizer,
                const CostFunction& costFn = CostFns::CrossEntropy())
      : NeuralNetwork(costFn, optmizer) {}

  NeuralNetwork(const NeuralNetwork& nn);

  // add layer
  void add(const Layer& layer);

  // upload weights from csv file.
  void uploadWeights(std::fstream& fin);

  Eigen::MatrixXd printResults(
      const Eigen::MatrixXd& inputData, const Eigen::MatrixXd& expectedOutput,
      const double lambda = 0,
      double (*accuracyFn)(const Eigen::VectorXd& outputData,
                           const Eigen::VectorXd& expectedOutput) =
          getAccuracy) const;

  Eigen::MatrixXd allOutputs(const Eigen::MatrixXd& inputData) const;

  double calcTotalCost(const Eigen::MatrixXd& outputData,
                       const Eigen::MatrixXd& expectedOutputData,
                       const double lambda) const;

  static double calcCost(const Eigen::VectorXd& outputData,
                         const Eigen::VectorXd& expectedOutput);

  static double calcSingleCost(double expected, double output);

  std::vector<std::vector<double>> train(
      const Eigen::MatrixXd& inputData, const Eigen::MatrixXd& outputData,
      const double lambda, const int batchSize, const int totalRounds,
      const bool printWeightsAndLastChange, const int costRecordInterval);

  std::vector<std::vector<double>> train(const Eigen::MatrixXd& inputData,
                                         const Eigen::MatrixXd& outputData,
                                         const int batchSize,
                                         const int totalRounds,
                                         const double lambda = 0) {
    return train(inputData, outputData, lambda, batchSize, totalRounds, false,
                 1000);
  }

  std::vector<Eigen::MatrixXd> getDecBy(
      const Eigen::VectorXd& inputData, const Eigen::VectorXd& outputData,
      const double lambda, std::vector<Eigen::MatrixXd> decBy) const;

  void printWeightsAndUpdates(std::vector<Eigen::MatrixXd> decBy) const;

  void printWeights() const;

  static double getTotalAccuracy(
      const Eigen::MatrixXd& inputData, const Eigen::MatrixXd& outputData,
      const Eigen::MatrixXd& expectedOutputData,
      double (*accuracyFn)(const Eigen::VectorXd& outputData,
                           const Eigen::VectorXd& expectedOutput));

  static double getAccuracy(const Eigen::VectorXd& outputData,
                            const Eigen::VectorXd& expectedOutputData);

  Eigen::MatrixXd getOutput(const Eigen::MatrixXd& inputData) const;

  std::unique_ptr<NeuralNetwork> clone() const;

  void putWeights(std::fstream& fout);

  std::vector<Eigen::MatrixXd> getWeights();

  std::vector<Eigen::MatrixXd> getWeights(std::fstream& fin);

  void assertInputAndOutputData(const Eigen::MatrixXd& inputData,
                                const Eigen::MatrixXd& outputData) const;

  void assertInputData(const Eigen::MatrixXd& inputData) const;

  void assertOutputData(const Eigen::MatrixXd& outputData) const;

  void assertLambda(const double lambda) const;
};

#endif