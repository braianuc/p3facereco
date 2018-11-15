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
// * Adapted version

package com.p3ds.facereco.java.facedetection;

import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;

import com.google.firebase.ml.vision.common.FirebaseVisionPoint;
import com.google.firebase.ml.vision.face.FirebaseVisionFace;
import com.google.firebase.ml.vision.face.FirebaseVisionFaceLandmark;
import com.p3ds.facereco.java.processing.CameraSource;
import com.p3ds.facereco.java.processing.GraphicOverlay;

/**
 * Graphic instance for rendering face position, orientation, and landmarks within an associated
 * graphic overlay view.
 */
public class FaceGraphic extends GraphicOverlay.Graphic {
    private static final float FACE_POSITION_RADIUS = 10.0f;
    private static final float ID_TEXT_SIZE = 40.0f;
    private static final float ID_Y_OFFSET = 50.0f;
    private static final float ID_X_OFFSET = -100.0f;
    private static final float BOX_STROKE_WIDTH = 8.0f;

    private static final int[] COLOR_CHOICES = {
            Color.YELLOW
            //Color.BLUE, Color.CYAN, Color.GREEN, Color.MAGENTA, Color.RED, Color.WHITE, Color.YELLOW
    };
    private static int currentColorIndex = 0;

    private int cameraFacing;

    private final Paint facePositionPaint;
    private final Paint idPaint;
    private final Paint boxPaint;

    private volatile FirebaseVisionFace firebaseVisionFace;

    private String faceName = null;

    public FaceGraphic(GraphicOverlay overlay) {
        super(overlay);

        currentColorIndex = (currentColorIndex + 1) % COLOR_CHOICES.length;
        final int selectedColor = COLOR_CHOICES[currentColorIndex];

        facePositionPaint = new Paint();
        facePositionPaint.setColor(selectedColor);

        idPaint = new Paint();
        idPaint.setColor(selectedColor);
        idPaint.setTextSize(ID_TEXT_SIZE);

        boxPaint = new Paint();
        boxPaint.setColor(selectedColor);
        boxPaint.setStyle(Paint.Style.STROKE);
        boxPaint.setStrokeWidth(BOX_STROKE_WIDTH);
    }

    /**
     * Updates the face instance from the detection of the most recent frame. Invalidates the relevant
     * portions of the overlay to trigger a redraw.
     */
    public void updateFace(FirebaseVisionFace face, int facing, String faceName) {
        firebaseVisionFace = face;
        this.cameraFacing = facing;
        this.faceName = faceName;
        postInvalidate();
    }

    /**
     * Draws the face annotations for position on the supplied canvas.
     */
    @Override
    public void draw(Canvas canvas) {
        if (null == firebaseVisionFace)
            return;
        drawRectangle(firebaseVisionFace, canvas);
    }

    /**
     * Draw a rectangle around the face.
     */
    private void drawRectangle(FirebaseVisionFace face, Canvas canvas) {
        FaceBounds faceBounds = getFaceBoundsForFace(face);
        canvas.drawRect(faceBounds.getLeft(), faceBounds.getTop(), faceBounds.getRight(), faceBounds.getBottom(), boxPaint);
        if (null != faceName) {
            //canvas.drawText("Face ID: " + face.getTrackingId(), x + ID_X_OFFSET, bottom + ID_Y_OFFSET, idPaint);
            canvas.drawText(faceName, faceBounds.getX() + ID_X_OFFSET, faceBounds.getBottom() + ID_Y_OFFSET, idPaint);
        }
        float x = faceBounds.getX();
        float y = faceBounds.getY();


        //<editor-fold>
    canvas.drawCircle(x, y, FACE_POSITION_RADIUS, facePositionPaint);

    /*
    canvas.drawText(
        "happiness: " + String.format("%.2f", face.getSmilingProbability()),
        x + ID_X_OFFSET * 6,
        y - ID_Y_OFFSET,
        idPaint);
    if (cameraFacing == CameraSource.CAMERA_FACING_FRONT) {
      canvas.drawText(
          "right eye: " + String.format("%.2f", face.getRightEyeOpenProbability()),
          x - ID_X_OFFSET,
          y,
          idPaint);
      canvas.drawText(
          "left eye: " + String.format("%.2f", face.getLeftEyeOpenProbability()),
          x + ID_X_OFFSET * 6,
          y,
          idPaint);
    } else {
      canvas.drawText(
          "left eye: " + String.format("%.2f", face.getLeftEyeOpenProbability()),
          x - ID_X_OFFSET,
          y,
          idPaint);
      canvas.drawText(
          "right eye: " + String.format("%.2f", face.getRightEyeOpenProbability()),
          x + ID_X_OFFSET * 6,
          y,
          idPaint);
    }*/


        // draw landmarks
    drawLandmarkPosition(canvas, face, FirebaseVisionFaceLandmark.BOTTOM_MOUTH);
    drawLandmarkPosition(canvas, face, FirebaseVisionFaceLandmark.LEFT_CHEEK);
    drawLandmarkPosition(canvas, face, FirebaseVisionFaceLandmark.LEFT_EAR);
    drawLandmarkPosition(canvas, face, FirebaseVisionFaceLandmark.LEFT_MOUTH);
    drawLandmarkPosition(canvas, face, FirebaseVisionFaceLandmark.LEFT_EYE);
    drawLandmarkPosition(canvas, face, FirebaseVisionFaceLandmark.NOSE_BASE);
    drawLandmarkPosition(canvas, face, FirebaseVisionFaceLandmark.RIGHT_CHEEK);
    drawLandmarkPosition(canvas, face, FirebaseVisionFaceLandmark.RIGHT_EAR);
    drawLandmarkPosition(canvas, face, FirebaseVisionFaceLandmark.RIGHT_EYE);
    drawLandmarkPosition(canvas, face, FirebaseVisionFaceLandmark.RIGHT_MOUTH);

        //</editor-fold>



    }

    public FaceBounds getFaceBoundsForFace(FirebaseVisionFace face) {
        float x = translateX(face.getBoundingBox().centerX());
        float y = translateY(face.getBoundingBox().centerY());
        float xOffset = scaleX(face.getBoundingBox().width() / 2.0f);
        float yOffset = scaleY(face.getBoundingBox().height() / 2.0f);
        float left = x - xOffset;
        float top = y - yOffset;
        float right = x + xOffset;
        float bottom = y + yOffset;
        return new FaceBounds(x, y, xOffset, yOffset, left, top, right, bottom);
    }

    public static class FaceBounds {

        private float x;
        private float y;
        private float xOffset;
        private float yOffset;

        private float left;
        private float top;
        private float bottom;
        private float right;

        public FaceBounds(float x, float y, float xOffset, float yOffset, float left, float top, float right, float bottom) {
            this.x = x;
            this.y = y;
            this.xOffset = xOffset;
            this.yOffset = yOffset;
            this.left = left;
            this.top = top;
            this.bottom = bottom;
            this.right = right;
        }

        public float getLeft() {
            return left;
        }

        public float getTop() {
            return top;
        }

        public float getBottom() {
            return bottom;
        }

        public float getRight() {
            return right;
        }

        public float getX() {
            return x;
        }

        public float getY() {
            return y;
        }

        public float getxOffset() {
            return xOffset;
        }

        public float getyOffset() {
            return yOffset;
        }


        public int getWidth() {
            return (int) (right - left);
        }

        public int getHeight() {
            return (int) (bottom - top);
        }

        public int getWidthOffset(int x) {
            return (int) (x - left);
        }

        public int getHeightOffset(int y) {
            return (int) (y - top);
        }

    }


    //<editor-fold>

  private void drawLandmarkPosition(Canvas canvas, FirebaseVisionFace face, int landmarkID) {
    FirebaseVisionFaceLandmark landmark = face.getLandmark(landmarkID);
    if (landmark != null) {
      FirebaseVisionPoint point = landmark.getPosition();
      canvas.drawCircle(
              translateX(point.getX()),
              translateY(point.getY()),
              10f, idPaint);
    }
  }
    //</editor-fold>
}
