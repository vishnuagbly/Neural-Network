#ifndef NEURAL_NETWORK_H
#define NEURAL_NETWORK_H
#pragma once
#include <bits/stdc++.h>

#include <Eigen/Dense>

#include "CostFunction.h"
#include "Layer.h"
#include "Optimizer.h"
void printLayersValues(std::vector<Eigen::VectorXd> layersValues);

class NeuralNetwork {
  std::vector<std::unique_ptr<Layer>> LayersProps;
  int expectedOutputSize;
  //   Eigen::VectorXd (*calcOutputErr)(const Eigen::VectorXd& outputData,
  //                                    const Eigen::VectorXd& lastLayer);
  std::unique_ptr<CostFunction> costFn;
  std::unique_ptr<Optimizer<Eigen::MatrixXd>> optimizer;

  int totalLayers() { return LayersProps.size(); }

  std::vector<Eigen::VectorXd> calcAllNodes(const Eigen::VectorXd& inputData);

  std::vector<Eigen::VectorXd> calcDell(
      const Eigen::VectorXd& outputData,
      const std::vector<Eigen::VectorXd>& layersValues);

  std::vector<Eigen::MatrixXd> calcBigDell(
      const std::vector<Eigen::VectorXd>& dell,
      const std::vector<Eigen::VectorXd>& layersValues);

  std::vector<Eigen::MatrixXd> updateDecBy(
      const std::vector<Eigen::MatrixXd>& allBigDellValues,
      const std::vector<Eigen::VectorXd>& layersValues, const double lambda,
      std::vector<Eigen::MatrixXd> decBy);

  std::vector<Eigen::MatrixXd> updateWeights(std::vector<Eigen::MatrixXd> decBy,
                                             const double totalInputs);

 public:
  NeuralNetwork(const std::vector<int>& layerSizes,
                std::vector<Eigen::MatrixXd> weights,
                int expectedOutputSize = -1,
                const CostFunction& costFn = CostFns::CrossEntropy(),
                const Activation& activation = Activations::Sigmoid(),
                const Optimizer<Eigen::MatrixXd>& optimizer =
                    Optimizers::SGD<Eigen::MatrixXd>());

  NeuralNetwork(const std::vector<int>& layerSizes,
                std::vector<Eigen::MatrixXd> weights,
                const Activation& activation,
                const Optimizer<Eigen::MatrixXd>& optimizer =
                    Optimizers::SGD<Eigen::MatrixXd>())
      : NeuralNetwork(layerSizes, weights, -1, CostFns::CrossEntropy(),
                      activation, optimizer) {}

  NeuralNetwork(const std::vector<int>& layerSizes,
                std::vector<Eigen::MatrixXd> weights,
                const Optimizer<Eigen::MatrixXd>& optimizer =
                    Optimizers::SGD<Eigen::MatrixXd>(),
                const Activation& activation = Activations::Sigmoid())
      : NeuralNetwork(layerSizes, weights, activation, optimizer) {}

  NeuralNetwork(std::vector<std::unique_ptr<Layer>>& layers,
                const Optimizer<Eigen::MatrixXd>& optimizer =
                    Optimizers::SGD<Eigen::MatrixXd>(),
                const CostFunction& costFn = CostFns::CrossEntropy());

  NeuralNetwork(std::vector<std::unique_ptr<Layer>>& layers,
                const CostFunction& costFn,
                const Optimizer<Eigen::MatrixXd>& optimizer =
                    Optimizers::SGD<Eigen::MatrixXd>())
      : NeuralNetwork(layers, optimizer, costFn) {}

  static Eigen::VectorXd calcOutputErrDefaultFn(
      const Eigen::VectorXd& outputData, const Eigen::VectorXd& lastLayer);

  Eigen::MatrixXd printResults(
      const Eigen::MatrixXd& inputData, const Eigen::MatrixXd& expectedOutput,
      const double lambda,
      double (*accuracyFn)(const Eigen::VectorXd& outputData,
                           const Eigen::VectorXd& expectedOutput) = getAccuracy,
      const CostFunction& = CostFns::CrossEntropy());

  Eigen::MatrixXd allOutputs(const Eigen::MatrixXd& inputData);

  double calcTotalCost(const Eigen::MatrixXd& outputData,
                       const Eigen::MatrixXd& expectedOutputData,
                       const double lambda,
                       const CostFunction& = CostFns::CrossEntropy());

  static double calcCost(const Eigen::VectorXd& outputData,
                         const Eigen::VectorXd& expectedOutput);

  static double calcSingleCost(double expected, double output);

  std::vector<std::vector<double>> trainNetwork(
      const Eigen::MatrixXd& inputData, const Eigen::MatrixXd& outputData,
      const double lambda, const int batchSize, const int totalRounds,
      const bool printWeightsAndLastChange = false,
      const int costRecordInterval = 1000,
      const CostFunction& costFn = CostFns::CrossEntropy());

  std::vector<std::vector<double>> trainNetwork(
      const Eigen::MatrixXd& inputData, const Eigen::MatrixXd& outputData,
      const double lambda, const int batchSize, const int totalRounds,
      const CostFunction& costFn) {
    return trainNetwork(inputData, outputData, lambda, batchSize, totalRounds,
                        false, 1000, costFn);
  }

  std::vector<Eigen::MatrixXd> getDecBy(const Eigen::VectorXd& inputData,
                                        const Eigen::VectorXd& outputData,
                                        const double lambda,
                                        std::vector<Eigen::MatrixXd> decBy);

  void printWeightsAndUpdates(std::vector<Eigen::MatrixXd> decBy);

  void printWeights();

  static double getTotalAccuracy(
      const Eigen::MatrixXd& inputData, const Eigen::MatrixXd& outputData,
      const Eigen::MatrixXd& expectedOutputData,
      double (*accuracyFn)(const Eigen::VectorXd& outputData,
                           const Eigen::VectorXd& expectedOutput));

  static double getAccuracy(const Eigen::VectorXd& outputData,
                            const Eigen::VectorXd& expectedOutputData);

  Eigen::VectorXd getOutput(const Eigen::VectorXd& inputData);

  void assertInputAndOutputData(const Eigen::MatrixXd& inputData,
                                const Eigen::MatrixXd& outputData);

  void assertInputData(const Eigen::MatrixXd& inputData);

  void assertOutputData(const Eigen::MatrixXd& outputData);

  void assertLambda(const double lambda);
};

#endif