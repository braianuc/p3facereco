// Copyright 2018 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
package com.google.firebase.samples.apps.mlkit.java.facedetection;

import android.app.Activity;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.graphics.ImageFormat;
import android.graphics.Rect;
import android.graphics.YuvImage;
import android.os.Environment;
import android.support.annotation.NonNull;
import android.util.Log;

import com.google.android.gms.tasks.Task;
import com.google.firebase.ml.vision.FirebaseVision;
import com.google.firebase.ml.vision.common.FirebaseVisionImage;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetector;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceDetectorOptions;
import com.google.firebase.samples.apps.mlkit.common.FrameMetadata;
import com.google.firebase.samples.apps.mlkit.common.GraphicOverlay;
import com.google.firebase.samples.apps.mlkit.java.VisionProcessorBase;
import com.google.firebase.samples.apps.mlkit.java.facerecognition.FaceRecognitionProcessor;

import java.io.ByteArrayOutputStream;
import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.util.List;

/** Face Detector Demo. */
public class FaceDetectionProcessor extends VisionProcessorBase<List<FirebaseVisionFace>> {

  private static final String TAG = "FaceDetectionProcessor";

  private final FirebaseVisionFaceDetector detector;
  private FaceRecognitionProcessor processor;

  private Bitmap bitmap = null;

  public FaceDetectionProcessor(Activity livePreviewActivity) throws IOException {
    FirebaseVisionFaceDetectorOptions options =
        new FirebaseVisionFaceDetectorOptions.Builder()
            .setClassificationType(FirebaseVisionFaceDetectorOptions.ALL_CLASSIFICATIONS)
            .setLandmarkType(FirebaseVisionFaceDetectorOptions.ALL_LANDMARKS)
            .setTrackingEnabled(true)
            .build();

    detector = FirebaseVision.getInstance().getVisionFaceDetector(options);

    //System.out.println("ACTIVITY ASSETS");
    //System.err.println(livePreviewActivity.getAssets());
    //System.err.println(livePreviewActivity.getAssets().open("emp.txt").toString());
    //System.out.println(livePreviewActivity.getAssets().openFd("emp.tflite");
    processor = new FaceRecognitionProcessor(livePreviewActivity);
  }

  @Override
  public void stop() {
    try {
      detector.close();
      processor.close();
    } catch (IOException e) {
      Log.e(TAG, "Exception thrown while trying to close Face Detector: " + e);
    }
  }

  @Override
  protected Task<List<FirebaseVisionFace>> detectInImage(FirebaseVisionImage image) {
    return detector.detectInImage(image);
  }

  private int frameCount = 0;

  @Override
  protected void onSuccess(
          FirebaseVisionImage image,
      @NonNull List<FirebaseVisionFace> faces,
      @NonNull FrameMetadata frameMetadata,
      @NonNull GraphicOverlay graphicOverlay) {
    graphicOverlay.clear();
    faces.forEach(face -> {
        String result = null;
        FaceGraphic faceGraphic = new FaceGraphic(graphicOverlay);
        graphicOverlay.add(faceGraphic);
        if(null != bitmap) {
          // TODO crop bitmap
          frameCount++;
          if(frameCount == 100) {
            FaceGraphic.FaceBounds bounds = faceGraphic.getFaceBoundsForFace(face);
            int x = (int) bounds.getLeft() < 0 ? 0 : (int) bounds.getLeft();
            int y = (int) bounds.getTop() > bitmap.getHeight() ? bitmap.getHeight() : (int) bounds.getTop();
            int width = bounds.getWidth() - 150;
            int height = bounds.getHeight() - 100;
            if(x + width > bitmap.getWidth()) {
                width = bitmap.getWidth() -  x;
            }
              if(y + height > bitmap.getHeight()) {
                  height = bitmap.getHeight() - y;
              }
            System.err.printf(String.format("\n%s %s %s %s\n", x, y, width, height));
            Bitmap croppedFaceBmp = Bitmap.createBitmap(bitmap, x, y, width, height);
            String path = Environment.getExternalStorageDirectory().toString() + "/croppedFaceBmpxx" + face.getTrackingId() + ".jpg";
            try (FileOutputStream out = new FileOutputStream(new File(path))) {
              croppedFaceBmp.compress(Bitmap.CompressFormat.PNG, 100, out);
              System.err.println("Saved Bitmap to " + path);
            } catch (IOException e) {
              e.printStackTrace();
            }
          }
          result = processor.classifyFrame(bitmap);
        }
      faceGraphic.updateFace(face, frameMetadata.getCameraFacing(), result);
    });
  }

  @Override
  protected void onFailure(@NonNull Exception e) {
    Log.e(TAG, "Face detection failed " + e);
  }


  @Override
  public void process(ByteBuffer data, FrameMetadata frameMetadata, GraphicOverlay graphicOverlay) {
    super.process(data, frameMetadata, graphicOverlay);
    data.order(ByteOrder.nativeOrder());
    bitmap = createResizedBitmap(data, frameMetadata.getWidth(), frameMetadata.getHeight());
  }


  /** Resizes image data from {@code ByteBuffer}. */
  private Bitmap createResizedBitmap(ByteBuffer buffer, int width, int height) {
    YuvImage img = new YuvImage(buffer.array(), ImageFormat.NV21, width, height, null);
    ByteArrayOutputStream out = new ByteArrayOutputStream();
    img.compressToJpeg(new Rect(0, 0, img.getWidth(), img.getHeight()), 50, out);
    byte[] imageBytes = out.toByteArray();
    return BitmapFactory.decodeByteArray(imageBytes, 0, imageBytes.length);
  }

}
