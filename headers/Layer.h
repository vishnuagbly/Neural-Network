#ifndef LAYER_H
#define LAYER_H
#include <Eigen/Dense>
#include <memory>

#include "ActivationFunction.h"
#include "KernelInitializer.h"

class Layer {
 protected:
  int units;
  static Eigen::MatrixXd getKernel(const Eigen::MatrixXd& matrix);
  static Eigen::MatrixXd getBias(const Eigen::MatrixXd& matrix);

 public:
  Layer(int units);
  virtual ~Layer() {}
  virtual std::string getName() const = 0;
  virtual void setInputSize(int inputSize) = 0;
  virtual void initializeWeights(int inputSize) = 0;
  virtual void initializeWeights(int inputSize,
                                 const Eigen::MatrixXd& weights) = 0;
  virtual void updateWeights(const Eigen::MatrixXd& updatedWeights) = 0;
  virtual Eigen::VectorXd getZValues(
      const Eigen::VectorXd& lastLayer) const = 0;
  virtual Eigen::VectorXd getOutputValues(
      const Eigen::VectorXd& lastLayer) const = 0;
  virtual Eigen::VectorXd backPropagateDell(
      const Eigen::VectorXd& thisLayerDell) const = 0;
  virtual Eigen::MatrixXd calcBigDell(const Eigen::VectorXd& thisLayerDell,
                                      const Eigen::VectorXd& lastLayer,
                                      const double lambda) const = 0;
  int size() const;
  virtual Eigen::MatrixXd getWeights() const = 0;
  virtual Activation* getActivation() const = 0;
  virtual std::unique_ptr<Layer> clone() const = 0;
};

namespace Layers {
class Dense : public Layer {
  int inputSize;
  Eigen::MatrixXd weights;
  std::unique_ptr<KernelInitializer> initializer;
  std::unique_ptr<Activation> activation;

 public:
  Dense(const Dense* denseLayer);
  Dense(int units, Eigen::MatrixXd weights,
        const Activation& activation = Activations::Linear());
  Dense(int units, int inputSize, Eigen::MatrixXd weights,
        const Activation& activation = Activations::Linear());
  Dense(int units, const KernelInitializer& initializer,
        const Activation& activation);
  Dense(int units, const Activation& activation,
        const KernelInitializer& initializer)
      : Dense(units, initializer, activation) {}
  Dense(int units, const Activation& activation)
      : Dense(units, Initializers::GlorotNormal(), activation) {}
  Dense(int units, const KernelInitializer& initializer)
      : Dense(units, initializer, Activations::Linear()) {}
  explicit Dense(int units) : Dense(units, Activations::Linear()) {}

  static const std::string name;
  std::string getName() const;
  void setInputSize(int inputSize);
  void initializeWeights(int inputSize);
  void initializeWeights(int inputSize, const Eigen::MatrixXd& weights);
  void updateWeights(const Eigen::MatrixXd& updatedWeights);
  Eigen::VectorXd getZValues(const Eigen::VectorXd& lastLayer) const;
  Eigen::VectorXd getOutputValues(const Eigen::VectorXd& lastLayer) const;
  Eigen::VectorXd backPropagateDell(const Eigen::VectorXd& thisLayerDell) const;
  Eigen::MatrixXd calcBigDell(const Eigen::VectorXd& thisLayerDell,
                              const Eigen::VectorXd& lastLayer,
                              const double lambda) const;
  Eigen::MatrixXd getWeights() const;
  Activation* getActivation() const;
  int getInputSize() const;
  std::unique_ptr<Activation> getActivationObj() const;
  std::unique_ptr<KernelInitializer> getInitializerObj() const;
  std::unique_ptr<Layer> clone() const;
};

class Input : public Layer {
 public:
  static const std::string name;
  std::string getName() const;
  Input(int size);
  void setInputSize(int inputSize);
  void initializeWeights(int inputSize);
  void initializeWeights(int inputSize, const Eigen::MatrixXd& weights);
  void updateWeights(const Eigen::MatrixXd& updatedWeights);
  Eigen::VectorXd getZValues(const Eigen::VectorXd& lastLayer) const;
  Eigen::VectorXd getOutputValues(const Eigen::VectorXd& lastLayer) const;
  Eigen::VectorXd backPropagateDell(const Eigen::VectorXd& thisLayerDell) const;
  Eigen::MatrixXd calcBigDell(const Eigen::VectorXd& thisLayerDell,
                              const Eigen::VectorXd& lastLayer,
                              const double lambda) const;
  Eigen::MatrixXd getWeights() const;
  Activation* getActivation() const;
  std::unique_ptr<Layer> clone() const;
};

class Linear : public Layer {
  Eigen::MatrixXd weights;
  std::unique_ptr<Activation> activation;
  std::unique_ptr<KernelInitializer> initializer;

 public:
  static const std::string name;
  std::string getName() const;
  Linear(const Linear* linear);
  Linear(const Activation& activation, const KernelInitializer& initializer);
  Linear(const Activation& activation)
      : Linear(activation, Initializers::Ones()) {}
  Linear(const KernelInitializer& initializer)
      : Linear(Activations::Linear(), initializer) {}
  virtual ~Linear() {}
  void setInputSize(int inputSize);
  void initializeWeights(int inputSize);
  void initializeWeights(int inputSize, const Eigen::MatrixXd& weights);
  void updateWeights(const Eigen::MatrixXd& updatedWeights);
  Eigen::VectorXd getZValues(const Eigen::VectorXd& lastLayer) const;
  Eigen::VectorXd getOutputValues(const Eigen::VectorXd& lastLayer) const;
  Eigen::VectorXd backPropagateDell(const Eigen::VectorXd& thisLayerDell) const;
  Eigen::MatrixXd calcBigDell(const Eigen::VectorXd& thisLayerDell,
                              const Eigen::VectorXd& lastLayer,
                              const double lambda) const;
  Eigen::MatrixXd getWeights() const;
  Activation* getActivation() const;
  std::unique_ptr<Layer> clone() const;
};

class ConstLinear : public Linear {
 public:
  static const std::string name;
  ConstLinear(const Activation& activation,
              const KernelInitializer& initializer)
      : Linear(activation, initializer) {}
  ConstLinear(const Activation& activation)
      : Linear(activation, Initializers::Ones()) {}
  ConstLinear(const KernelInitializer& initializer)
      : Linear(Activations::Linear(), initializer) {}
  void updateWeights(const Eigen::MatrixXd& updatedWeights);
};

}  // namespace Layers

#endif