import TensorFlow

public struct CSPBlock<T: TensorFlowFloatingPoint>: Layer {
    public var base: ConvBN<T>
    public var part1: ConvBN<T>
    public var part2: ConvBN<T>
    public var denseBlock: [ShortcutBlock<T>]
    public var transition1: ConvBN<T>
    public var transition2: ConvBN<T>

    public init(inFilters: Int, outFilters: Int, repeatDense: Int = 1, expand: Bool = false){
      denseBlock = []
      precondition(inFilters * 2 == outFilters, "inFilters * 2 == outFilters")
      if (!expand) {   
          base  = ConvBN<T>(inFilters: inFilters, outFilters: outFilters, kernelSize: 3, stride: 2)
          part1 = ConvBN<T>(inFilters: outFilters, outFilters: inFilters, kernelSize: 1)
          part2 = ConvBN<T>(inFilters: outFilters, outFilters: inFilters, kernelSize: 1)
          for _ in 1...repeatDense {
              denseBlock.append(ShortcutBlock<T>(inFilters: inFilters, outFilters: inFilters, middleFilters: inFilters))
          }
          transition1 = ConvBN<T>(inFilters: inFilters, outFilters: inFilters, kernelSize: 1)
          transition2 = ConvBN<T>(inFilters: outFilters, outFilters: outFilters, kernelSize: 1)
      } else {
          base  = ConvBN<T>(inFilters: inFilters, outFilters: outFilters, kernelSize: 3, stride: 2)
          part1 = ConvBN<T>(inFilters: outFilters, outFilters: outFilters, kernelSize: 1)
          part2 = ConvBN<T>(inFilters: outFilters, outFilters: outFilters, kernelSize: 1)
          for _ in 1...repeatDense {
              denseBlock.append(ShortcutBlock<T>(inFilters: outFilters, outFilters: outFilters, middleFilters: inFilters))
          }
          transition1 = ConvBN<T>(inFilters: outFilters, outFilters: outFilters, kernelSize: 1)
          transition2 = ConvBN<T>(inFilters: 2*outFilters, outFilters: outFilters, kernelSize: 1)
      }
    }

    @differentiable
    public func callAsFunction(_ input: Tensor<Float>) -> Tensor<Float> {
        let baseResult = input |> base
        let route1Result = baseResult |> part1
        let route2Result = baseResult |> part2 |> denseBlock |> transition1
        let concate = route2Result.concatenated(with: route1Result, alongAxis: 3)
        let output = concate |> transition2
        return output
    }
}
