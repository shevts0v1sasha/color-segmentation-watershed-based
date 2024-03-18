package ru.hypercore.util;

import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.imgproc.Imgproc;

import java.util.List;

public final class CvUtils {
    private CvUtils() {
    }

    public static Mat bilateralFilter(Mat src, int epoch, int sigmaColor, int sigmaSpace) {
        Mat bilaterial = new Mat();
        Imgproc.bilateralFilter(src, bilaterial, -1, sigmaColor, sigmaSpace);

        for (int i = 1; i < epoch; i++) {
            Mat bilaterialOld = bilaterial.clone();
            bilaterial.release();
            bilaterial = new Mat();
            Imgproc.bilateralFilter(bilaterialOld, bilaterial, -1, sigmaColor, sigmaSpace);
        }

        return bilaterial;
    }
}
