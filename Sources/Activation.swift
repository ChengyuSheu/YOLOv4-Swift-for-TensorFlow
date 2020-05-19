import TensorFlow

@differentiable(wrt: x)
public func myLeakyRelu<T: TensorFlowFloatingPoint>(
  _ x: Tensor<T>
) -> Tensor<T> {
    leakyRelu(x, alpha: 0.1)
}

@differentiable
public func mish<T: TensorFlowFloatingPoint>(_ features: Tensor<T>) -> Tensor<T> {
  features * tanh(softplus(features))
}
