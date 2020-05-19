// swift-tools-version:5.2
// The swift-tools-version declares the minimum version of Swift required to build this package.

import PackageDescription

let package = Package(
    name: "YOLOv4",
    products: [
        .library(name: "YOLOv4", targets: ["YOLOv4"]),
    ],
    dependencies: [],
    targets: [
        .target(
            name: "YOLOv4",
            path: "Sources"),
    ]
)
