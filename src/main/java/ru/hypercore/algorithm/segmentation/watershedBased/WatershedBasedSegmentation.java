package ru.hypercore.algorithm.segmentation.watershedBased;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;
import ru.hypercore.util.CvUtils;
import ru.hypercore.util.Pair;

import java.awt.*;
import java.io.File;
import java.util.*;
import java.util.List;

public class WatershedBasedSegmentation {

    static {
        System.load(
                System.getProperty("user.dir") + File.separator +
                        "libs" + File.separator +
                        "opencv" + File.separator +
                        "opencv_java455.dll"
        );
    }

    private long ID = 0;

    public Mat analyze(Mat src, int watershedSeedPointsGap, double colorThresh, int bilateralSigmaColor, int bilateralSigmaSpace) {
        List<Pair<Point, double[]>> markerAndColorList = new ArrayList<>();
        Mat bilateral = CvUtils.bilateralFilter(src, 1, bilateralSigmaColor, bilateralSigmaSpace);
        Mat watershedResultRgba = watershedImageToRGBA(src, markerAndColorList, watershedSeedPointsGap, bilateral);
        List<FloodFillArea> areas = areas(markerAndColorList, watershedResultRgba);

        for (var area : areas) {
            double[] meanColor = {0, 0, 0};
            var points = area.floodFillResult().pointsInside();
            for (var p : points) {
                var doubles = bilateral.get((int) p.y, (int) p.x);
                meanColor[0] += doubles[0];
                meanColor[1] += doubles[1];
                meanColor[2] += doubles[2];
            }

            meanColor[0] /= points.size();
            meanColor[1] /= points.size();
            meanColor[2] /= points.size();

            area.color()[0] = meanColor[0];
            area.color()[1] = meanColor[1];
            area.color()[2] = meanColor[2];
        }

        Mat srcAreas = Mat.zeros(src.size(), CvType.CV_8UC3);
        areas.forEach(area -> area.floodFillResult().pointsInside().forEach(p -> srcAreas.put((int) p.y, (int) p.x, area.color())));
        mergeAreasByColor(areas, colorThresh);
        Mat res = Mat.zeros(src.size(), CvType.CV_8UC3);
        for (var area : areas) {
            var color = area.color();
            area.floodFillResult().pointsInside().forEach(p -> res.put((int) p.y, (int) p.x, color));
        }

        for (int row = 1; row < res.rows() - 1; row++) {
            for (int col = 0; col < res.cols(); col++) {
                var doubles_1 = res.get(row - 1, col);
                var doubles_2 = watershedResultRgba.get(row, col);
                var doubles_3 = res.get(row + 1, col);

                if (Arrays.equals(doubles_1, doubles_3) && doubles_2[3] == 0) {
                    res.put(row, col, doubles_1);
                }

            }
        }

        for (int row = 0; row < res.rows(); row++) {
            for (int col = 1; col < res.cols() - 1; col++) {
                var doubles_1 = res.get(row, col - 1);
                var doubles_2 = watershedResultRgba.get(row, col);
                var doubles_3 = res.get(row, col + 1);

                if (Arrays.equals(doubles_1, doubles_3) && doubles_2[3] == 0) {
                    res.put(row, col, doubles_1);
                }
            }
        }

        return res;
    }

    private Mat watershedImageToRGBA(Mat src, List<Pair<Point, double[]>> markerList, int watershedPadding, Mat bilateral) {
        Mat result = Mat.zeros(src.size(), CvType.CV_8UC3);
        Imgproc.cvtColor(result, result, Imgproc.COLOR_BGR2BGRA);
        Mat markers = Mat.zeros(src.size(), CvType.CV_32SC1);

        int markerColor = 1;
        for (int row = watershedPadding / 2; row < src.rows() - 1; row += watershedPadding) {
            for (int col = watershedPadding / 2; col < src.cols() - 1; col += watershedPadding) {
                markers.put(row, col, markerColor++);
                var doubles = bilateral.get(row, col);
                markerList.add(new Pair<>(new org.opencv.core.Point(col, row), doubles));
            }
        }

        Imgproc.watershed(bilateral, markers);
        for (int row = 0; row < src.rows(); row++) {
            for (int col = 0; col < src.cols(); col++) {
                if (markers.get(row, col)[0] > 0) {
                    result.put(row, col, 255, 255, 255, 255);
                } else {
                    result.put(row, col, 255, 255, 255, 0);
                }
            }
        }

        return result;
    }

    private List<FloodFillArea> areas(List<Pair<Point, double[]>> markerAndColorList, Mat watershedResult) {
        List<FloodFillArea> areas = new ArrayList<>();

        Map<Point, Set<Long>> neighbors = new HashMap<>();
        Map<Long, FloodFillArea> areasByIdMap = new HashMap<>();

        for (var markerAndColor : markerAndColorList) {
            FloodFillResult floodFillResult = myFloodFill(watershedResult, markerAndColor.key(), neighbors);
            var e = new FloodFillArea(
                    floodFillResult.id(),
                    markerAndColor.key(),
                    markerAndColor.value(),
                    new HashSet<>(),
                    floodFillResult
            );
            areasByIdMap.put(e.id(), e);
            areas.add(e);

        }


        // detect neighbors
        for (var entry : neighbors.entrySet()) {
            var neighborsIds = entry.getValue();
            for (var id : neighborsIds) {
                var area = areasByIdMap.get(id);
                for (var neighborId : neighborsIds) {
                    if (id.equals(neighborId)) continue;
                    var neighbor = areasByIdMap.get(neighborId);
                    area.addNeighbor(neighbor);
                    neighbor.addNeighbor(area);
                }
            }
        }

        return areas;
    }

    private static void mergeAreasByColor(List<FloodFillArea> areas, double colorThresh) {
        boolean allMerged = false;

        m:
        while (!allMerged) {
            for (var area : areas) {
                var color = area.color();
                var neighbors = area.neighbors();

                for (var neighbor : neighbors) {

                    var neighborColor = neighbor.color();
                    var colorsEquals = compareColors(color, neighborColor, colorThresh);

                    if (colorsEquals) {

                        neighbors.remove(neighbor);
                        areas.remove(neighbor);

                        var neighborNeighbors = neighbor.neighbors();
                        neighborNeighbors.remove(area);
                        neighborNeighbors.forEach(floodFillArea -> floodFillArea.neighbors().remove(neighbor));
                        neighborNeighbors.forEach(floodFillArea -> floodFillArea.neighbors().add(area));
                        area.floodFillResult().pointsInside().addAll(neighbor.floodFillResult().pointsInside());
                        area.neighbors().addAll(neighborNeighbors);

                        continue m;
                    }
                }
            }
            allMerged = true;
        }
    }

    public static boolean compareColors(double[] color_1, double[] color_2, double thresh) {
        if (color_1 == null || color_2 == null) return false;
        double r1 = color_1[2], g1 = color_1[1], b1 = color_1[0];
        double r2 = color_2[2], g2 = color_2[1], b2 = color_2[0];

        float[] hsv_1 = new float[3];
        float[] hsv_2 = new float[3];
        Color.RGBtoHSB((int) r1, (int) g1, (int) b1, hsv_1);
        Color.RGBtoHSB((int) r2, (int) g2, (int) b2, hsv_2);
        hsv_1[0] *= 360;
        hsv_1[1] *= 255;
        hsv_1[2] *= 255;

        hsv_2[0] *= 360;
        hsv_2[1] *= 255;
        hsv_2[2] *= 255;

        double dh = Math.min(hsv_2[0] - hsv_1[0], 360 - Math.abs(hsv_2[0] - hsv_1[0]));
        double ds = Math.abs(hsv_2[1] - hsv_1[1]);
        double dv = Math.abs(hsv_2[2] - hsv_1[2]);
        double distance = Math.sqrt(dh * dh + ds * ds + dv * dv);

        return distance < thresh;
    }

    private FloodFillResult myFloodFill(Mat mat, Point seedPoint, Map<Point, Set<Long>> neighbors) {

        long id = ID++;

        var imageHeight = mat.rows();
        var imageWidth = mat.cols();

        Set<Point> visitedPixels = new HashSet<>();
        Set<Point> borders = new HashSet<>();

        double[] seedPointColor = mat.get((int) seedPoint.y, (int) seedPoint.x);

        Stack<Point> stack = new Stack<>();
        stack.push(seedPoint);

        while (!stack.isEmpty()) {
            var currentPixel = stack.pop();
            int currentCol = (int) currentPixel.x;
            int currentRow = (int) currentPixel.y;

            visitedPixels.add(currentPixel);

//            LEFT
            var leftPoint = new Point(currentCol - 1, currentRow);
            if (!visitedPixels.contains(leftPoint)) {
                if (currentRow >= 0 && currentRow < imageHeight && currentCol - 1 >= 0 && currentCol - 1 < imageWidth) {
                    var color = mat.get((int) leftPoint.y, (int) leftPoint.x);
                    if (Arrays.equals(seedPointColor, color)) {
                        stack.push(leftPoint);
                    } else {
                        borders.add(leftPoint);
                        if (neighbors.containsKey(leftPoint)) {
                            neighbors.get(leftPoint).add(id);
                        } else {
                            neighbors.put(leftPoint, new HashSet<>() {{
                                add(id);
                            }});
                        }
                    }
                }
            }

            //            TOP
            var topPoint = new Point(currentCol, currentRow - 1);
            if (!visitedPixels.contains(topPoint)) {
                if (currentRow - 1 >= 0 && currentRow - 1 < imageHeight && currentCol >= 0 && currentCol < imageWidth) {
                    var color = mat.get((int) topPoint.y, (int) topPoint.x);
                    if (Arrays.equals(seedPointColor, color)) {
                        stack.push(topPoint);
                    } else {
                        borders.add(topPoint);
                        if (neighbors.containsKey(topPoint)) {
                            neighbors.get(topPoint).add(id);
                        } else {
                            neighbors.put(topPoint, new HashSet<>() {{
                                add(id);
                            }});
                        }
                    }
                }
            }

            //            RIGHT
            var rightPoint = new Point(currentCol + 1, currentRow);
            if (!visitedPixels.contains(rightPoint)) {
                if (currentRow >= 0 && currentRow < imageHeight && currentCol + 1 >= 0 && currentCol + 1 < imageWidth) {
                    var color = mat.get((int) rightPoint.y, (int) rightPoint.x);
                    if (Arrays.equals(seedPointColor, color)) {
                        stack.push(rightPoint);
                    } else {
                        borders.add(rightPoint);
                        if (neighbors.containsKey(rightPoint)) {
                            neighbors.get(rightPoint).add(id);
                        } else {
                            neighbors.put(rightPoint, new HashSet<>() {{
                                add(id);
                            }});
                        }
                    }
                }
            }

            //            BOTTOM
            var bottomPoint = new Point(currentCol, currentRow + 1);
            if (!visitedPixels.contains(bottomPoint)) {
                if (currentRow + 1 >= 0 && currentRow + 1 < imageHeight && currentCol >= 0 && currentCol < imageWidth) {
                    var color = mat.get((int) bottomPoint.y, (int) bottomPoint.x);
                    if (Arrays.equals(seedPointColor, color)) {
                        stack.push(bottomPoint);
                    } else {
                        borders.add(bottomPoint);
                        if (neighbors.containsKey(bottomPoint)) {
                            neighbors.get(bottomPoint).add(id);
                        } else {
                            neighbors.put(bottomPoint, new HashSet<>() {{
                                add(id);
                            }});
                        }
                    }
                }
            }
        }

        return new FloodFillResult(id, visitedPixels, borders);
    }
}
