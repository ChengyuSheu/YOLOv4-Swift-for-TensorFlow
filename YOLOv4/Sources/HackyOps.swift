import TensorFlow
import Foundation

precedencegroup ForwardFlow {
  associativity: left
}

infix operator |> : ForwardFlow

@differentiable
public func |> <Features: Differentiable, Func: Layer>(left: Func.Input, right:Func) -> Func.Output where Features == Func.Input {
  return right(left);
}


public func getWeights <T: TensorFlowFloatingPoint> (
    _ stream: InputStream,
    _ size: Int) -> Tensor<T> {
            var buf:[UInt8] = [UInt8](repeating: 0, count:  size * MemoryLayout<T>.size)
            var f: [T] = [T](repeating: 0, count: size)
            var len = stream.read(&buf, maxLength: buf.count)
            memcpy(&f, buf, len)
            var weights = Tensor<T>(shape:[size], scalars: f)
    return weights
}

extension Array: Module where Element: Layer, Element.Input == Element.Output {
  public typealias Input = Element.Input
  public typealias Output = Element.Output

  @differentiable
  public func callAsFunction(_ input: Input) -> Output {
    differentiableReduce(input) { $1($0) }
  }
}

extension Array: Layer where Element: Layer, Element.Input == Element.Output {}

import Dispatch
public func time(_ function: () -> ()) {
    let start = DispatchTime.now()
    function()
    let end = DispatchTime.now()
    let nanoseconds = Double(end.uptimeNanoseconds - start.uptimeNanoseconds)
    let milliseconds = nanoseconds / 1e6
    // Could do fancier pretty-printing logic based on the order of magnitude.
    print("\(milliseconds) ms")
}

public func time(re: Int, _ function: () -> ()) {
    time{
          for i in 1...re {
            function()
          }
        }
}

@usableFromInline
@differentiable(wrt: images where Scalar: TensorFlowFloatingPoint)
func resizeNearestNeighbor<Scalar: TensorFlowNumeric>(
  images: Tensor<Scalar>,
  size: Tensor<Int32>,
  alignCorners: Bool = true,
  halfPixelCenters: Bool = false
) -> Tensor<Scalar> {
  _Raw.resizeNearestNeighbor(
    images: images,
    size: size,
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters
  )
}

@usableFromInline
@derivative(of: resizeNearestNeighbor)
func _vjpResizeNearestNeighbor<Scalar: TensorFlowFloatingPoint>(
  images: Tensor<Scalar>,
  size: Tensor<Int32>,
  alignCorners: Bool,
  halfPixelCenters: Bool
) -> (value: Tensor<Scalar>, pullback: (Tensor<Scalar>) -> Tensor<Scalar>) {
  let resized = resizeNearestNeighbor(
    images: images,
    size: size,
    alignCorners: alignCorners,
    halfPixelCenters: halfPixelCenters
  )
  return (
    resized,
    { v in
      _Raw.resizeNearestNeighborGrad(
        grads: v,
        size: Tensor([Int32(images.shape[1]), Int32(images.shape[2])], on: .defaultTFEager),
        alignCorners: alignCorners,
        halfPixelCenters: halfPixelCenters
      )
    }
  )
}


