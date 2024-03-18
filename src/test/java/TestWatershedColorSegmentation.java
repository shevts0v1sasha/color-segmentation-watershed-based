import junit.framework.TestCase;
import org.opencv.core.Mat;
import org.opencv.imgcodecs.Imgcodecs;
import ru.hypercore.algorithm.segmentation.watershedBased.WatershedBasedSegmentation;

import java.io.File;

public class TestWatershedColorSegmentation extends TestCase {

    static {
        System.load(
                System.getProperty("user.dir") + File.separator +
                        "libs" + File.separator +
                        "opencv" + File.separator +
                        "opencv_java455.dll"
        );
    }

    public void testWatershedColorSegmentation() {
        File imgFile = new File((System.getProperty("user.dir") + "/src/main/resources/img/src.jpg").replace("\\", "/"));
        Mat src = Imgcodecs.imread(imgFile.getAbsolutePath());

        if (src.empty()) {
            throw new RuntimeException("Src img with path \"%s\" is empty".formatted(imgFile.getAbsolutePath()));
        }

        WatershedBasedSegmentation segmentation = new WatershedBasedSegmentation();
        var result = segmentation.analyze(src, 20, 20, 50, 30);

        if (result == null || result.empty()) {
            throw new RuntimeException("Result of watershed color segmentation is null");
        }

        Imgcodecs.imwrite(imgFile.getParentFile().getAbsolutePath() + "/result.jpg", result);
    }
}
