import TensorFlow
public struct Bbox {
    public var x1: Float
    public var x2: Float
    public var y1: Float
    public var y2: Float
    public var score: Float
    public var cls: [Float]
    init (
        values: [Float] = [0,0,0,0],
        score: Float = 1,
        cls: [Float] = []
    )
    {
        x1 = min(values[0], values[1])
        x2 = max(values[0], values[1])
        y1 = min(values[2], values[3])
        y2 = max(values[2], values[3])
        self.score = score
        self.cls = cls
    }
}

public func & (A: Bbox, B: Bbox) -> Float {
    let l = max(A.x1, B.x1)
    let r = min(A.x2, B.x2)
    let t = max(A.y1, B.y1)
    let d = min(A.y2, B.y2)
    let area = (d-t) * (r-l)
    return max(area, 0)
}

public func | (A: Bbox, B: Bbox) -> Float {
    A.area() + B.area() - (A & B)
}

public func * (A: Bbox, B: Bbox) -> Float {
    (A & B) / (A | B)
}

extension Bbox: Comparable {
    public static func == (A: Bbox, B:Bbox) -> Bool {
        A.score == B.score
    }
    public static func < (A: Bbox, B:Bbox) -> Bool {
        A.score < B.score
    }
    public static func > (A: Bbox, B:Bbox) -> Bool {
        A.score > B.score
    }
    public static func <= (A: Bbox, B:Bbox) -> Bool {
        A.score <= B.score
    }
    public static func >= (A: Bbox, B:Bbox) -> Bool {
        A.score >= B.score
    }

}

extension Bbox {
    public func area() -> Float {
        (x2 - x1) * (y2 - y1)
    }
}

public func argsort<T:Comparable>( _ a : [T] ) -> [Int] {
    var r = Array(a.indices)
    r.sort(by: { a[$0] < a[$1] })
    return r
}

public func load(from data: Tensor<Float>, anc: Int, at: Int) -> Bbox {
        var shape = data[0, anc, 0...3, at].scalars
        var cls = data[0, anc, 5...(data.shape[2]), at].scalars
        var obj = data[0, anc, 4, at].scalar!
        return Bbox(values: shape, score: obj, cls: cls)
}

public func load(from tensor: Tensor<Float>) -> [Bbox] {
        var t = tensor.reshaped(to: [1, 3, 85, tensor.shape[2]])
        var res: [Bbox] = []
        for anc in 0...2 {
            for i in 0...t.shape[3]-1 {
                var x = load(from: t, anc: anc, at: i)
                res.append(x)
            }
        }
        return res
}

public func nms(bboxes: [Bbox], thresh: Float = 0.25, thresh_iou: Float = 0.25) -> [Bbox] {
    var boxes = bboxes.filter{$0.score > thresh}
    var idxs = argsort(boxes)
    var dets: [Bbox] = []
    while (idxs.count > 0) {
        var last = idxs.count - 1
        dets.append(boxes[idxs.last!])
        var ious = idxs.prefix(last).map{($0, boxes[$0] * boxes[idxs.last!])}
        idxs = ious.filter{ $0.1 < thresh_iou}.map{ $0.0 }
    }
    return dets
}