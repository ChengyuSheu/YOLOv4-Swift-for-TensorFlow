import TensorFlow

public struct MyUpSampling2D<Scalar: TensorFlowFloatingPoint>: ParameterlessLayer {
    @noDerivative public let size: Int
      public init(size: Int) {
        self.size = size
      }
    
    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        
        let newSize = Tensor([Int32(input.shape[1]*size), Int32(input.shape[2]*size)], on: .defaultTFEager)
        var x: Tensor<Float> =  resizeNearestNeighbor(images: input, size: newSize)
        return x
        
    }
}