#ifndef LAYER_H
#define LAYER_H
#include <Eigen/Dense>
#include <memory>

#include "ActivationFunction.h"
#include "KernelInitializer.h"

class Layer {
  int units;

 public:
  Layer(int units);
  virtual std::string getName() const = 0;
  virtual void setInputSize(int inputSize) = 0;
  virtual void initializeWeights(int inputSize) = 0;
  virtual void initializeWeights(int inputSize, Eigen::MatrixXd weights) = 0;
  virtual void updateWeights(Eigen::MatrixXd updatedWeights) = 0;
  int size() const;
  virtual Eigen::MatrixXd getWeights() const = 0;
  virtual Activation* getActivation() = 0;
  virtual std::unique_ptr<Layer> getLayer() const = 0;
};

namespace Layers {
class DenseLayer : public Layer {
  int inputSize;
  Eigen::MatrixXd weights;
  std::unique_ptr<KernelInitializer> initializer;
  std::unique_ptr<Activation> activation;

 public:
  DenseLayer(const DenseLayer* denseLayer);
  DenseLayer(int units, Eigen::MatrixXd weights,
             const Activation& activation = Activations::Sigmoid());
  DenseLayer(int units, int inputSize, Eigen::MatrixXd weights,
             const Activation& activation = Activations::Sigmoid());
  DenseLayer(int units, const KernelInitializer& initializer,
             const Activation& activation);
  DenseLayer(int units, const Activation& activation,
             const KernelInitializer& initializer)
      : DenseLayer(units, initializer, activation) {}
  DenseLayer(int units, const Activation& activation)
      : DenseLayer(units, Initializers::GlorotNormal(), activation) {}
  DenseLayer(int units, const KernelInitializer& initializer)
      : DenseLayer(units, initializer, Activations::Sigmoid()) {}
  explicit DenseLayer(int units) : DenseLayer(units, Activations::Sigmoid()) {}

  static const std::string name;
  std::string getName() const;
  void setInputSize(int inputSize);
  void initializeWeights(int inputSize);
  void initializeWeights(int inputSize, Eigen::MatrixXd weights);
  void updateWeights(Eigen::MatrixXd updatedWeights);
  Eigen::MatrixXd getWeights() const;
  Activation* getActivation();
  int getInputSize() const;
  std::unique_ptr<Activation> getActivationObj() const;
  std::unique_ptr<KernelInitializer> getInitializerObj() const;
  std::unique_ptr<Layer> getLayer() const;
};

class InputLayer : public Layer {
 public:
  static const std::string name;
  std::string getName() const;
  InputLayer(int size);
  void setInputSize(int inputSize);
  void initializeWeights(int inputSize);
  void initializeWeights(int inputSize, Eigen::MatrixXd weights);
  void updateWeights(Eigen::MatrixXd updatedWeights);
  Eigen::MatrixXd getWeights() const;
  Activation* getActivation();
  std::unique_ptr<Layer> getLayer() const;
};
}  // namespace Layers

#endif