import Accelerate
import CoreImage
import Foundation
import UIKit

import onnxruntime_mobile_objc

final class OnnxCardDetector {
  static let shared = OnnxCardDetector()

  private let inputSize: Int = 640
  private let confThreshold: Float = 0.02
  private let iouThreshold: Float = 0.45
  private let maxDetections: Int = 50

  private let ciContext = CIContext(options: nil)
  private let sessionQueue = DispatchQueue(label: "card_detector.onnx.session")

  private var labels: [String] = []
  private var env: ORTEnv?
  private var session: ORTSession?
  private var inputName: String?
  private var outputName: String?
  private var lastOutputShape: [Int] = []
  private var lastMaxScore: Float = 0
  private var lastMaxLabel: String = ""
  private var lastCandidates: Int = 0
  private var lastSelected: Int = 0

  private init() {}

  func debugInfo() -> [String: Any] {
    sessionQueue.sync {
      [
        "configured": session != nil,
        "labels": labels.count,
        "inputName": inputName ?? "",
        "outputName": outputName ?? "",
        "outputShape": lastOutputShape,
        "maxScore": lastMaxScore,
        "maxLabel": lastMaxLabel,
        "candidates": lastCandidates,
        "selected": lastSelected,
      ]
    }
  }

  func configure(modelPath: String, labelsPath: String) throws {
    try sessionQueue.sync {
      guard FileManager.default.fileExists(atPath: modelPath) else {
        throw NSError(domain: "card_detector", code: 1, userInfo: [NSLocalizedDescriptionKey: "Model not found at path."])
      }
      guard FileManager.default.fileExists(atPath: labelsPath) else {
        throw NSError(domain: "card_detector", code: 2, userInfo: [NSLocalizedDescriptionKey: "Labels not found at path."])
      }

      let raw = try String(contentsOfFile: labelsPath, encoding: .utf8)
      labels = raw
        .split(whereSeparator: \.isNewline)
        .map { String($0).trimmingCharacters(in: .whitespacesAndNewlines) }
        .filter { !$0.isEmpty }

      let env = try ORTEnv(loggingLevel: ORTLoggingLevel.warning)

      let options = try ORTSessionOptions()
      _ = try? options.setGraphOptimizationLevel(ORTGraphOptimizationLevel.all)
      _ = try? options.setIntraOpNumThreads(2)
      _ = try? options.appendExecutionProvider("coreml", providerOptions: [:])
      _ = try? options.appendExecutionProvider("xnnpack", providerOptions: [:])

      let session = try ORTSession(env: env, modelPath: modelPath, sessionOptions: options)

      let inputNames = try session.inputNames()
      let outputNames = try session.outputNames()

      guard let inputName = inputNames.first, let outputName = outputNames.first else {
        throw NSError(domain: "card_detector", code: 5, userInfo: [NSLocalizedDescriptionKey: "Model input/output names missing."])
      }

      self.env = env
      self.session = session
      self.inputName = inputName
      self.outputName = outputName
      self.lastOutputShape = []
      self.lastMaxScore = 0
      self.lastMaxLabel = ""
      self.lastCandidates = 0
      self.lastSelected = 0
    }
  }

  func detect(pixelBuffer: CVPixelBuffer, viewSize: CGSize) -> [Detection] {
    sessionQueue.sync {
      guard let session, let inputName, let outputName else { return [] }

      guard let cgImage = cgImageFrom(pixelBuffer: pixelBuffer) else { return [] }
      // Use the oriented CGImage size for all downstream geometry.
      let sourceSize = CGSize(width: cgImage.width, height: cgImage.height)
      guard let prep = preprocess(cgImage: cgImage, sourceSize: sourceSize) else { return [] }

      guard let output = run(session: session, inputName: inputName, outputName: outputName, inputTensor: prep.tensor) else {
        return []
      }

      let decoded = decodeYolo(
        output: output,
        labels: labels,
        confThreshold: confThreshold,
        iouThreshold: iouThreshold,
        maxDetections: maxDetections
      )
      lastMaxScore = decoded.maxScore
      lastMaxLabel = decoded.maxLabel
      lastCandidates = decoded.candidates
      lastSelected = decoded.detections.count
      return decoded.detections.map { d in
        let modelRect = d.rect
        let origRect = prep.letterbox.unletterbox(modelRect: modelRect, originalSize: sourceSize)
        let viewRect = mapToAspectFillView(rectInSource: origRect, sourceSize: sourceSize, viewSize: viewSize)
        return Detection(label: d.label, confidence: Double(d.score), bbox: viewRect)
      }
    }
  }

  func detect(image: UIImage) -> [Detection] {
    sessionQueue.sync {
      guard let session, let inputName, let outputName else { return [] }
      guard let cgImage = image.cgImage else { return [] }

      let sourceSize = CGSize(width: cgImage.width, height: cgImage.height)
      guard let prep = preprocess(cgImage: cgImage, sourceSize: sourceSize) else { return [] }

      guard let output = run(session: session, inputName: inputName, outputName: outputName, inputTensor: prep.tensor) else {
        return []
      }

      let decoded = decodeYolo(
        output: output,
        labels: labels,
        confThreshold: confThreshold,
        iouThreshold: iouThreshold,
        maxDetections: maxDetections
      )
      lastMaxScore = decoded.maxScore
      lastMaxLabel = decoded.maxLabel
      lastCandidates = decoded.candidates
      lastSelected = decoded.detections.count
      return decoded.detections.map { d in
        let modelRect = d.rect
        let origRect = prep.letterbox.unletterbox(modelRect: modelRect, originalSize: sourceSize)
        let norm = CGRect(
          x: max(0, min(1, origRect.minX / sourceSize.width)),
          y: max(0, min(1, origRect.minY / sourceSize.height)),
          width: max(0, min(1, origRect.width / sourceSize.width)),
          height: max(0, min(1, origRect.height / sourceSize.height))
        )
        return Detection(label: d.label, confidence: Double(d.score), bbox: norm)
      }
    }
  }

  struct OutputTensor {
    let values: [Float]
    let shape: [Int]
  }

  private func run(session: ORTSession, inputName: String, outputName: String, inputTensor: NSMutableData) -> OutputTensor? {
    guard let input = try? ORTValue(
      tensorData: inputTensor,
      elementType: ORTTensorElementDataType.float,
      shape: [1, 3, NSNumber(value: inputSize), NSNumber(value: inputSize)]
    ) else { return nil }

    do {
      let outputs = try session.run(withInputs: [inputName: input], outputNames: Set([outputName]), runOptions: nil)
      guard let out = outputs[outputName] else { return nil }

      let shape: [Int]
      if let info = try? out.tensorTypeAndShapeInfo() {
        shape = info.shape.map { $0.intValue }
      } else {
        shape = []
      }
      if !shape.isEmpty, shape != lastOutputShape {
        lastOutputShape = shape
        NSLog("[card_detector] ONNX output shape: \(shape)")
      }

      let data = try out.tensorData()
      let count = data.length / MemoryLayout<Float>.size
      let base = data.bytes.assumingMemoryBound(to: Float.self)
      let values = Array(UnsafeBufferPointer(start: base, count: count))
      return OutputTensor(values: values, shape: shape)
    } catch {
      NSLog("[card_detector] ONNX run failed: \(error)")
      return nil
    }
  }

  private func cgImageFrom(pixelBuffer: CVPixelBuffer) -> CGImage? {
    let ciImage = CIImage(cvPixelBuffer: pixelBuffer)
    // Best-effort: assume portrait orientation; adjust if needed during tuning.
    let oriented = ciImage.oriented(.right)
    return ciContext.createCGImage(oriented, from: oriented.extent)
  }

  private struct PreprocessResult {
    let tensor: NSMutableData
    let letterbox: Letterbox
  }

  private func preprocess(cgImage: CGImage, sourceSize: CGSize) -> PreprocessResult? {
    let lb = Letterbox(input: CGFloat(inputSize), original: sourceSize)

    let w = inputSize
    let h = inputSize
    let bytesPerRow = w * 4
    var rgba = [UInt8](repeating: 0, count: w * h * 4)
    // Fill with 114 (Ultralytics default) and opaque alpha.
    for i in stride(from: 0, to: rgba.count, by: 4) {
      rgba[i] = 114
      rgba[i + 1] = 114
      rgba[i + 2] = 114
      rgba[i + 3] = 255
    }

    rgba.withUnsafeMutableBytes { ptr in
      guard let ctx = CGContext(
        data: ptr.baseAddress,
        width: w,
        height: h,
        bitsPerComponent: 8,
        bytesPerRow: bytesPerRow,
        space: CGColorSpaceCreateDeviceRGB(),
        bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue
      ) else { return }
      ctx.interpolationQuality = .high
      ctx.draw(cgImage, in: lb.drawRect)
    }

    // Convert RGBA -> Float32 CHW RGB normalized [0,1]
    let floatCount = 3 * w * h
    let floats = NSMutableData(length: floatCount * MemoryLayout<Float>.size)!
    let outBase = floats.mutableBytes.assumingMemoryBound(to: Float.self)
    var idxR = 0
    var idxG = w * h
    var idxB = 2 * w * h
    for i in stride(from: 0, to: rgba.count, by: 4) {
      outBase[idxR] = Float(rgba[i]) / 255.0
      outBase[idxG] = Float(rgba[i + 1]) / 255.0
      outBase[idxB] = Float(rgba[i + 2]) / 255.0
      idxR += 1
      idxG += 1
      idxB += 1
    }

    return PreprocessResult(tensor: floats, letterbox: lb)
  }
}

private struct Letterbox {
  let input: CGFloat
  let original: CGSize
  let scale: CGFloat
  let padX: CGFloat
  let padY: CGFloat
  let drawRect: CGRect

  init(input: CGFloat, original: CGSize) {
    self.input = input
    self.original = original

    let r = min(input / original.width, input / original.height)
    scale = r
    let newW = original.width * r
    let newH = original.height * r
    padX = (input - newW) / 2
    padY = (input - newH) / 2
    drawRect = CGRect(x: padX, y: padY, width: newW, height: newH)
  }

  func unletterbox(modelRect: CGRect, originalSize: CGSize) -> CGRect {
    let x1 = (modelRect.minX - padX) / scale
    let y1 = (modelRect.minY - padY) / scale
    let x2 = (modelRect.maxX - padX) / scale
    let y2 = (modelRect.maxY - padY) / scale

    let clampedX1 = max(0, min(originalSize.width, x1))
    let clampedY1 = max(0, min(originalSize.height, y1))
    let clampedX2 = max(0, min(originalSize.width, x2))
    let clampedY2 = max(0, min(originalSize.height, y2))
    return CGRect(x: clampedX1, y: clampedY1, width: max(0, clampedX2 - clampedX1), height: max(0, clampedY2 - clampedY1))
  }
}

private struct DecodedDetection {
  let rect: CGRect   // model-space (0..input)
  let score: Float
  let classIndex: Int
  let label: String
}

private struct DecodeResult {
  let detections: [DecodedDetection]
  let maxScore: Float
  let maxLabel: String
  let candidates: Int
}

private func decodeYolo(
  output: OnnxCardDetector.OutputTensor,
  labels: [String],
  confThreshold: Float,
  iouThreshold: Float,
  maxDetections: Int
) -> DecodeResult {
  let values = output.values
  let nc = max(labels.count, 1)

  // Accept common shapes:
  // - [1, 4+nc, N] or [1, 5+nc, N]
  // - [1, N, 4+nc] or [1, N, 5+nc]
  // Fallback: infer by divisibility.
  var channels = 4 + nc
  var numPred = 0
  var channelMajor = true // values indexed as [c][i] contiguous by i
  var hasObjectness = false

  if output.shape.count == 3 {
    let d1 = output.shape[1]
    let d2 = output.shape[2]
    if d1 == 4 + nc || d1 == 5 + nc {
      channels = d1
      numPred = d2
      channelMajor = true
      hasObjectness = (d1 == 5 + nc)
    } else if d2 == 4 + nc || d2 == 5 + nc {
      channels = d2
      numPred = d1
      channelMajor = false
      hasObjectness = (d2 == 5 + nc)
    }
  }

  if numPred == 0 {
    // Fallback: try channels 4+nc and 5+nc.
    let c1 = 4 + nc
    let c2 = 5 + nc
    if values.count % c1 == 0 {
      channels = c1
      numPred = values.count / c1
      channelMajor = true
      hasObjectness = false
    } else if values.count % c2 == 0 {
      channels = c2
      numPred = values.count / c2
      channelMajor = true
      hasObjectness = true
    } else {
      return DecodeResult(detections: [], maxScore: 0, maxLabel: "", candidates: 0)
    }
  }

  func sigmoid(_ x: Float) -> Float { 1 / (1 + exp(-x)) }
  func prob(_ raw: Float) -> Float {
    // Some exports already output probabilities in [0,1]. If so, don't sigmoid again.
    if raw >= 0, raw <= 1 { return raw }
    return sigmoid(raw)
  }

  var candidates: [DecodedDetection] = []
  candidates.reserveCapacity(256)

  var maxScoreSeen: Float = 0
  var maxLabelSeen: String = ""

  for i in 0..<numPred {
    func v(_ c: Int) -> Float {
      if channelMajor {
        return values[i + c * numPred]
      } else {
        return values[c + i * channels]
      }
    }

    let x = v(0)
    let y = v(1)
    let w = v(2)
    let h = v(3)
    if w <= 0 || h <= 0 { continue }

    let obj: Float = hasObjectness ? prob(v(4)) : 1.0
    let clsOffset = hasObjectness ? 5 : 4

    var bestScore: Float = 0
    var bestClass: Int = -1
    for c in 0..<labels.count {
      let raw = v(clsOffset + c)
      let s = prob(raw) * obj
      if s > bestScore {
        bestScore = s
        bestClass = c
      }
    }

    if bestClass < 0 || bestScore < confThreshold { continue }

    let x1 = x - w / 2
    let y1 = y - h / 2
    let rect = CGRect(x: CGFloat(x1), y: CGFloat(y1), width: CGFloat(w), height: CGFloat(h))
    candidates.append(DecodedDetection(rect: rect, score: bestScore, classIndex: bestClass, label: labels[bestClass]))

    if bestScore > maxScoreSeen {
      maxScoreSeen = bestScore
      maxLabelSeen = labels[bestClass]
    }
  }

  if maxScoreSeen > 0 {
    NSLog("[card_detector] decode maxScore=\(maxScoreSeen) label=\(maxLabelSeen) candidates=\(candidates.count)")
  } else {
    NSLog("[card_detector] decode no candidates (confThreshold=\(confThreshold))")
  }

  let candidatesCount = candidates.count
  // Class-agnostic NMS.
  candidates.sort { $0.score > $1.score }
  var selected: [DecodedDetection] = []
  selected.reserveCapacity(min(maxDetections, candidates.count))

  for cand in candidates {
    var keep = true
    for sel in selected {
      if iou(a: cand.rect, b: sel.rect) > iouThreshold {
        keep = false
        break
      }
    }
    if keep {
      selected.append(cand)
      if selected.count >= maxDetections { break }
    }
  }
  return DecodeResult(detections: selected, maxScore: maxScoreSeen, maxLabel: maxLabelSeen, candidates: candidatesCount)
}

private func iou(a: CGRect, b: CGRect) -> Float {
  let inter = a.intersection(b)
  if inter.isNull { return 0 }
  let interArea = Float(inter.width * inter.height)
  let areaA = Float(a.width * a.height)
  let areaB = Float(b.width * b.height)
  return interArea / max(1e-6, (areaA + areaB - interArea))
}

private func mapToAspectFillView(rectInSource: CGRect, sourceSize: CGSize, viewSize: CGSize) -> CGRect {
  let sx = viewSize.width / sourceSize.width
  let sy = viewSize.height / sourceSize.height
  let scale = max(sx, sy)

  let displayedW = sourceSize.width * scale
  let displayedH = sourceSize.height * scale
  let offsetX = (viewSize.width - displayedW) / 2
  let offsetY = (viewSize.height - displayedH) / 2

  let x = rectInSource.minX * scale + offsetX
  let y = rectInSource.minY * scale + offsetY
  let w = rectInSource.width * scale
  let h = rectInSource.height * scale
  return CGRect(x: x, y: y, width: w, height: h)
}
