import { useCallback, useState } from "react";

export function useReceiptScanner() {
  const [isProcessing, setIsProcessing] = useState(false);

  const scanImage = useCallback((file) => {
    return new Promise((resolve, reject) => {
      setIsProcessing(true);

      const img = new Image();
      const objectUrl = URL.createObjectURL(file);

      img.onload = () => {
        let src = null;
        let gray = null;
        let blurred = null;
        let edges = null;
        let contours = null;
        let hierarchy = null;

        try {
          if (!window.cv) {
            throw new Error("OpenCV.js is not loaded.");
          }

          const cv = window.cv;

          src = cv.imread(img);

          /*
           * Work on a smaller image so contour detection is faster.
           * The original image is still used for the final crop.
           */
          const maxDimension = 1600;
          const scale =
            Math.max(src.rows, src.cols) > maxDimension
              ? maxDimension / Math.max(src.rows, src.cols)
              : 1;

          const processingWidth = Math.round(src.cols * scale);
          const processingHeight = Math.round(src.rows * scale);

          const resized = new cv.Mat();

          cv.resize(
            src,
            resized,
            new cv.Size(processingWidth, processingHeight),
          );

          gray = new cv.Mat();
          blurred = new cv.Mat();
          edges = new cv.Mat();
          contours = new cv.MatVector();
          hierarchy = new cv.Mat();

          cv.cvtColor(resized, gray, cv.COLOR_RGBA2GRAY);

          cv.GaussianBlur(
            gray,
            blurred,
            new cv.Size(5, 5),
            0,
            0,
            cv.BORDER_DEFAULT,
          );

          cv.Canny(blurred, edges, 30, 100);

          cv.findContours(
            edges,
            contours,
            hierarchy,
            cv.RETR_EXTERNAL,
            cv.CHAIN_APPROX_SIMPLE,
          );

          let bestContour = null;
          let bestArea = 0;
          let bestPoints = null;

          for (let i = 0; i < contours.size(); i++) {
            const contour = contours.get(i);

            const area = cv.contourArea(contour);

            if (area < 1000) {
              contour.delete();
              continue;
            }

            const perimeter = cv.arcLength(contour, true);

            const approximation = new cv.Mat();

            cv.approxPolyDP(contour, approximation, 0.02 * perimeter, true);

            if (approximation.rows >= 4 && area > bestArea) {
              bestArea = area;

              if (bestContour) {
                bestContour.delete();
              }

              if (bestPoints) {
                bestPoints.delete();
              }

              bestContour = contour;
              bestPoints = approximation;
            } else {
              approximation.delete();
              contour.delete();
            }
          }

          resized.delete();

          if (!bestContour) {
            throw new Error(
              `Receipt edges could not be detected. Image: ${src.cols}x${src.rows}`,
            );
          }

          const boundingRect = cv.boundingRect(bestContour);

          const points = [
            {
              x: boundingRect.x,
              y: boundingRect.y,
            },
            {
              x: boundingRect.x + boundingRect.width,
              y: boundingRect.y,
            },
            {
              x: boundingRect.x + boundingRect.width,
              y: boundingRect.y + boundingRect.height,
            },
            {
              x: boundingRect.x,
              y: boundingRect.y + boundingRect.height,
            },
          ];

          console.log("Detected receipt corners:", points);

          const orderedPoints = orderPoints(points);

          /*
           * Convert the detected points back to the
           * original image's coordinates.
           */
          const originalPoints = orderedPoints.map((point) => ({
            x: point.x / scale,
            y: point.y / scale,
          }));

          const topWidth = distance(originalPoints[0], originalPoints[1]);

          const bottomWidth = distance(originalPoints[2], originalPoints[3]);

          const leftHeight = distance(originalPoints[0], originalPoints[3]);

          const rightHeight = distance(originalPoints[1], originalPoints[2]);

          const outputWidth = Math.round(Math.max(topWidth, bottomWidth));

          const outputHeight = Math.round(Math.max(leftHeight, rightHeight));

          if (outputWidth < 100 || outputHeight < 100) {
            throw new Error("Detected receipt area is too small.");
          }

          const srcPoints = cv.matFromArray(
            4,
            1,
            cv.CV_32FC2,
            originalPoints.flatMap((point) => [point.x, point.y]),
          );

          const dstPoints = cv.matFromArray(4, 1, cv.CV_32FC2, [
            0,
            0,
            outputWidth - 1,
            0,
            outputWidth - 1,
            outputHeight - 1,
            0,
            outputHeight - 1,
          ]);

          const transform = cv.getPerspectiveTransform(srcPoints, dstPoints);

          const warped = new cv.Mat();

          cv.warpPerspective(
            src,
            warped,
            transform,
            new cv.Size(outputWidth, outputHeight),
          );

          const resultCanvas = document.createElement("canvas");

          cv.imshow(resultCanvas, warped);

          const dataUrl = resultCanvas.toDataURL("image/jpeg", 0.92);

          resolve(dataUrl);

          srcPoints.delete();
          dstPoints.delete();
          transform.delete();
          warped.delete();
          bestPoints.delete();

          if (bestContour) {
            bestContour.delete();
          }
        } catch (error) {
          console.error("Receipt detection failed:", error);

          reject(error);
        } finally {
          if (src) src.delete();
          if (gray) gray.delete();
          if (blurred) blurred.delete();
          if (edges) edges.delete();
          if (contours) contours.delete();
          if (hierarchy) hierarchy.delete();

          URL.revokeObjectURL(objectUrl);
          setIsProcessing(false);
        }
      };

      img.onerror = () => {
        URL.revokeObjectURL(objectUrl);
        setIsProcessing(false);

        reject(new Error("Could not load captured image."));
      };

      img.src = objectUrl;
    });
  }, []);

  return {
    scanImage,
    isProcessing,
  };
}

function distance(pointA, pointB) {
  return Math.hypot(pointA.x - pointB.x, pointA.y - pointB.y);
}

function orderPoints(points) {
  const sorted = [...points];

  const topLeft = sorted.reduce((best, point) =>
    point.x + point.y < best.x + best.y ? point : best,
  );

  const bottomRight = sorted.reduce((best, point) =>
    point.x + point.y > best.x + best.y ? point : best,
  );

  const topRight = sorted.reduce((best, point) =>
    point.x - point.y > best.x - best.y ? point : best,
  );

  const bottomLeft = sorted.reduce((best, point) =>
    point.x - point.y < best.x - best.y ? point : best,
  );

  return [topLeft, topRight, bottomRight, bottomLeft];
}
