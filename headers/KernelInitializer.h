#ifndef KERNELINITIALIZER_H
#define KERNELINITIALIZER_H
#include <Eigen/Dense>
#include <memory>

class KernelInitializer {
 public:
  virtual Eigen::MatrixXd generate(int lastLayerSize, int currentLayerSize) = 0;
  virtual std::unique_ptr<KernelInitializer> clone() const = 0;
};

namespace Initializers {
class HeNormal : public KernelInitializer {
 public:
  Eigen::MatrixXd generate(int lastLayerSize, int currentLayerSize);
  std::unique_ptr<KernelInitializer> clone() const;
};

class GlorotNormal : public KernelInitializer {
 public:
  Eigen::MatrixXd generate(int lastLayerSize, int currentLayerSize);
  std::unique_ptr<KernelInitializer> clone() const;
};

class Zero : public KernelInitializer {
 public:
  Eigen::MatrixXd generate(int lastLayerSize, int currentLayerSize);
  std::unique_ptr<KernelInitializer> clone() const;
};
}  // namespace Initializers

#endif