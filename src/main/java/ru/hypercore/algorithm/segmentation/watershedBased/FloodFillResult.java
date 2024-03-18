package ru.hypercore.algorithm.segmentation.watershedBased;

import org.opencv.core.Point;

import java.util.Set;

public record FloodFillResult(Long id, Set<Point> pointsInside, Set<Point> borders) {
}
