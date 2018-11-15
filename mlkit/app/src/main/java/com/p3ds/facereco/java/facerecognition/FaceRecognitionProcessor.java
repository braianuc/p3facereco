/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

package com.p3ds.facereco.java.facerecognition;

import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.res.AssetFileDescriptor;
import android.graphics.Bitmap;
import android.os.Environment;
import android.util.Log;

import org.tensorflow.lite.Interpreter;

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;
import java.util.AbstractMap;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.PriorityQueue;

/**
 * Recognize (classify) faces with TF Lite
 */
public class FaceRecognitionProcessor {

    /**
     * Tag for the {@link Log}.
     */
    private static final String TAG = "P3DSFaceReco";

    /**
     * Name of the model file stored in Assets.
     */
    private static final String MODEL_PATH = "emp.tflite";

    /**
     * Name of the label file stored in Assets.
     */
    private static final String LABEL_PATH = "retrained_labels.txt";

    /**
     * Number of results to show in the UI.
     */
    private static final int RESULTS_TO_SHOW = 3;

    /**
     * Dimensions of inputs.
     */
    private static final int DIM_BATCH_SIZE = 1;

    private static final int DIM_PIXEL_SIZE = 3;

    private static final int DIM_IMG_SIZE_X = 224;
    private static final int DIM_IMG_SIZE_Y = 224;

    private static final int IMAGE_MEAN = 128;
    private static final float IMAGE_STD = 128.0f;


    /**
     * An instance of the driver class to run model inference with Tensorflow Lite.
     */
    private Interpreter tfLite;

    /**
     * Labels corresponding to the output of the vision model.
     */
    private List<String> labelList;


    private static final int FILTER_STAGES = 3;
    private static final float FILTER_FACTOR = 0.4f;

    private PriorityQueue<Map.Entry<String, Float>> sortedLabels =
            new PriorityQueue<>(
                    RESULTS_TO_SHOW,
                    Comparator.comparing(o -> (o.getValue())));

    /**
     * Initializes an {@code ImageClassifier}.
     */
    public FaceRecognitionProcessor(Activity activity) throws IOException {
        tfLite = new Interpreter(loadModelFile(activity));
        labelList = loadLabelList(activity);
        Log.d(TAG, "Created a Tensorflow Lite Image Classifier.");
    }

    /**
     * Classifies a frame from the preview stream.
     */
    public Map.Entry<String, Float> classifyFrame(Bitmap bitmap) throws IOException {
        if (tfLite == null) {
            throw new IOException("Image classifier has not been initialized; Skipped.");
        }
        if (bitmap.isRecycled()) {
            throw new IOException("Bitmap recycled prematurely. Skip this frame.");
        }
        ByteBuffer imgData = convertBitmapToByteBuffer(bitmap);
        float[][] labelProb = applyFilter();
        tfLite.run(imgData, labelProb);
        bitmap.recycle();
        return getLabelAndConfidence(labelProb);
    }

    private float[][] applyFilter() {
        int num_labels = labelList.size();
        float[][] labelProbArray = new float[1][labelList.size()];
        float[][] filterLabelProbArray = new float[FILTER_STAGES][labelList.size()];
        for (int j = 0; j < num_labels; ++j) {
            filterLabelProbArray[0][j] += FILTER_FACTOR * (labelProbArray[0][j] -
                    filterLabelProbArray[0][j]);
        }
        for (int i = 1; i < FILTER_STAGES; ++i) {
            for (int j = 0; j < num_labels; ++j) {
                filterLabelProbArray[i][j] += FILTER_FACTOR * (
                        filterLabelProbArray[i - 1][j] -
                                filterLabelProbArray[i][j]);

            }
        }

        System.arraycopy(filterLabelProbArray[FILTER_STAGES - 1], 0, labelProbArray[0], 0, num_labels);
        return labelProbArray;
    }

    /**
     * Close TF Lite and release resources.
     */
    public void close() {
        tfLite.close();
        tfLite = null;
    }

    /**
     * Reads label list from Assets.
     */
    private List<String> loadLabelList(Activity activity) throws IOException {
        List<String> labelList = new ArrayList<>();
        BufferedReader reader =
                new BufferedReader(new InputStreamReader(activity.getAssets().open(LABEL_PATH)));
        String line;
        while ((line = reader.readLine()) != null) {
            labelList.add(line);
        }
        reader.close();
        return labelList;
    }

    /**
     * Memory-map the model file in Assets.
     */
    private MappedByteBuffer loadModelFile(Activity activity) throws IOException {
        AssetFileDescriptor fileDescriptor = activity.getAssets().openFd(MODEL_PATH);
        FileInputStream inputStream = new FileInputStream(fileDescriptor.getFileDescriptor());
        FileChannel fileChannel = inputStream.getChannel();
        long startOffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return fileChannel.map(FileChannel.MapMode.READ_ONLY, startOffset, declaredLength);
    }

    //int frameCount = 0;

    /**
     * Writes Image data into a {@code ByteBuffer}.
     */
    private ByteBuffer convertBitmapToByteBuffer(Bitmap bitmap) {
        ByteBuffer imgData = ByteBuffer.allocateDirect(4 * DIM_BATCH_SIZE * DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y * DIM_PIXEL_SIZE);
        imgData.order(ByteOrder.nativeOrder());
        imgData.rewind();
        bitmap = Bitmap.createScaledBitmap(bitmap, FaceRecognitionProcessor.DIM_IMG_SIZE_X, FaceRecognitionProcessor.DIM_IMG_SIZE_Y, true);
        // Save every 10th frame for debugging purposes

        //System.out.println("Frame " + frameCount);
        //frameCount++;
        //if(frameCount % 10 == 0) {
        //    try {
        //        saveBitmapToFile(bitmap, "faces/face" + ( frameCount ));
        //    } catch (IOException e) {
        //        e.printStackTrace();
        //    }
        //}
        int[] intValues = new int[DIM_IMG_SIZE_X * DIM_IMG_SIZE_Y];
        bitmap.getPixels(intValues, 0, bitmap.getWidth(), 0, 0, bitmap.getWidth(), bitmap.getHeight());
        int pixel = 0;
        for (int i = 0; i < DIM_IMG_SIZE_X; ++i) {
            for (int j = 0; j < DIM_IMG_SIZE_Y; ++j) {
                final int val = intValues[pixel++];
                imgData.putFloat((((val >> 16) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val >> 8) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
                imgData.putFloat((((val) & 0xFF) - IMAGE_MEAN) / IMAGE_STD);
            }
        }
        return imgData;
    }


    /**
     * Prints the label and its confidence level
     */
    @SuppressLint("DefaultLocale")
    private Map.Entry<String, Float> getLabelAndConfidence(float[][] labelProb) {
        for (int i = 0; i < labelList.size(); ++i) {
            sortedLabels.add(new AbstractMap.SimpleEntry<>(labelList.get(i), labelProb[0][i]));
            if (sortedLabels.size() > RESULTS_TO_SHOW) {
                sortedLabels.poll();
            }
        }
        //String textToShow = "";
        //final int size = sortedLabels.size();
        //for (int i = 0; i < size; ++i) {
        return sortedLabels.poll();
        //return String.format("\n%s (%.0f%%)", label.getKey().substring(0, 1).toUpperCase() + label.getKey().substring(1), label.getValue() * 100); // + textToShow
        //}
    }


    /**
     * Save a bitmap to file
     * @param bitmap input file
     * @param fileName output file name
     */
    @SuppressWarnings("unused")
    private void saveBitmapToFile(Bitmap bitmap, String fileName) throws IOException {
        String path = Environment.getExternalStorageDirectory().toString() + "/" + fileName + ".jpg";
        try (FileOutputStream out = new FileOutputStream(new File(path))) {
            if(bitmap.isRecycled()) {
                throw new IOException("Bitmap is recycled, cannot save.");
            }
            bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
            //System.out.println("Saved Bitmap to " + path);
        }
    }
}
