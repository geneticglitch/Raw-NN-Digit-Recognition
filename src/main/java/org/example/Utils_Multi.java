/*
    This class contains the methods that are parallelized using ForkJoinPool.
    Currently runs out of memory for large datasets.
    Will be updated in the future to use a more efficient parallelization method.
    Please refer to the Utils.java class for the sequential implementation.
 */

package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;
import java.util.concurrent.ForkJoinPool;
import java.util.concurrent.RecursiveTask;

public class Utils_Multi {

    private static final ForkJoinPool forkJoinPool = new ForkJoinPool();

    public static DataSet readCSV(String filePath) throws IOException {
        List<float[]> records = new ArrayList<>();
        try (Reader reader = new FileReader(filePath);
             CSVParser csvParser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
            for (CSVRecord csvRecord : csvParser) {
                int size = csvRecord.size();
                float[] row = new float[size];
                for (int i = 0; i < size; i++) {
                    try {
                        row[i] = Float.parseFloat(csvRecord.get(i));
                    } catch (NumberFormatException e) {
                        row[i] = Float.NaN;
                    }
                }
                records.add(row);
            }
        }

        int numRows = records.size();
        int numCols = records.get(0).length - 1;

        float[][] features = new float[numCols][numRows]; // Transpose: columns will become rows
        float[] labels = new float[numRows];

        for (int i = 0; i < numRows; i++) {
            float[] row = records.get(i);
            labels[i] = row[0];
            for (int j = 1; j <= numCols; j++) {
                features[j - 1][i] = (float) (row[j] / 255.0); // Normalize pixel values and transpose
            }
        }

        return new DataSet(features, labels);
    }

    public static float[][] ReLu(float[][] inputs) {
        return forkJoinPool.invoke(new ReLUTask(inputs, 0, inputs.length));
    }

    public static float[][] softmax(float[][] Z) {
        return forkJoinPool.invoke(new SoftmaxTask(Z, 0, Z.length));
    }

    public static float[][] dot_2d(float[][] A, float[][] B) {
        return forkJoinPool.invoke(new Dot2DTask(A, B, 0, A.length));
    }

    // Transpose
    public static float[][] transpose(float[][] matrix) {
        return forkJoinPool.invoke(new TransposeTask(matrix, 0, matrix.length));
    }

    // One-hot Encoding
    public static float[][] one_hot_encoding(float[] Y, int numClasses) {
        float[][] oneHot = new float[numClasses][Y.length];
        for (int i = 0; i < Y.length; i++) {
            int label = (int) Y[i];
            oneHot[label][i] = 1;
        }
        return oneHot;
    }

    // Subtract 1D
    public static float[] subtract_1d(float[] A, float[] B) {
        return forkJoinPool.invoke(new Subtract1DTask(A, B, 0, A.length));
    }

    // Subtract 2D
    public static float[][] subtract_2d(float[][] A, float[][] B) {
        return forkJoinPool.invoke(new Subtract2DTask(A, B, 0, A.length));
    }

    // Multiply Scalar 2D
    public static float[][] multiply_scalar_2d(float[][] A, float scalar) {
        return forkJoinPool.invoke(new MultiplyScalar2DTask(A, scalar, 0, A.length));
    }

    // Multiply Scalar 1D
    public static float[] multiply_scalar_1d(float[] A, float scalar) {
        return forkJoinPool.invoke(new MultiplyScalar1DTask(A, scalar, 0, A.length));
    }

    // Sum
    public static float[] sum(float[][] A) {
        return forkJoinPool.invoke(new SumTask(A, 0, A[0].length));
    }

    // Derivate ReLU
    public static float[][] derivate_ReLu(float[][] inputs) {
        return forkJoinPool.invoke(new DerivateReLUTask(inputs, 0, inputs.length));
    }

    // Multiply 2D
    public static float[][] multiply_2d(float[][] A, float[][] B) {
        return forkJoinPool.invoke(new Multiply2DTask(A, B, 0, A.length));
    }

    // Argmax
    public static int[] argmax(float[][] A) {
        return forkJoinPool.invoke(new ArgmaxTask(A, 0, A[0].length));
    }

    private static class ReLUTask extends RecursiveTask<float[][]> {
        private static final int THRESHOLD = 100;
        private final float[][] inputs;
        private final int start;
        private final int end;

        public ReLUTask(float[][] inputs, int start, int end) {
            this.inputs = inputs;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[][] compute() {
            if (end - start <= THRESHOLD) {
                float[][] outputs = new float[inputs.length][inputs[0].length];
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < inputs[0].length; j++) {
                        outputs[i][j] = Math.max(0, inputs[i][j]);
                    }
                }
                return outputs;
            } else {
                int mid = (start + end) / 2;
                ReLUTask leftTask = new ReLUTask(inputs, start, mid);
                ReLUTask rightTask = new ReLUTask(inputs, mid, end);
                invokeAll(leftTask, rightTask);
                float[][] leftResult = leftTask.join();
                float[][] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[][] combineResults(float[][] leftResult, float[][] rightResult) {
            float[][] result = new float[inputs.length][inputs[0].length];
            for (int i = 0; i < inputs.length; i++) {
                for (int j = 0; j < inputs[0].length; j++) {
                    if (i < leftResult.length) {
                        result[i][j] = leftResult[i][j];
                    } else {
                        result[i][j] = rightResult[i - leftResult.length][j];
                    }
                }
            }
            return result;
        }
    }

    private static class SoftmaxTask extends RecursiveTask<float[][]> {
        private static final int THRESHOLD = 100; // Threshold for task splitting
        private final float[][] Z;
        private final int start;
        private final int end;

        public SoftmaxTask(float[][] Z, int start, int end) {
            this.Z = Z;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[][] compute() {
            if (end - start <= THRESHOLD) {
                float[][] A = new float[Z.length][Z[0].length];
                float[][] expZ = new float[Z.length][Z[0].length];
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < Z[i].length; j++) {
                        expZ[i][j] = (float) Math.exp(Z[i][j]);
                    }
                }

                for (int j = 0; j < Z[0].length; j++) {
                    float sumExpZ = 0.0F;
                    for (int i = 0; i < Z.length; i++) {
                        sumExpZ += expZ[i][j];
                    }
                    for (int i = start; i < end; i++) {
                        A[i][j] = expZ[i][j] / sumExpZ;
                    }
                }

                return A;
            } else {
                int mid = (start + end) / 2;
                SoftmaxTask leftTask = new SoftmaxTask(Z, start, mid);
                SoftmaxTask rightTask = new SoftmaxTask(Z, mid, end);
                invokeAll(leftTask, rightTask);
                float[][] leftResult = leftTask.join();
                float[][] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[][] combineResults(float[][] leftResult, float[][] rightResult) {
            float[][] result = new float[Z.length][Z[0].length];
            for (int i = 0; i < Z.length; i++) {
                for (int j = 0; j < Z[0].length; j++) {
                    if (i < leftResult.length) {
                        result[i][j] = leftResult[i][j];
                    } else {
                        result[i][j] = rightResult[i - leftResult.length][j];
                    }
                }
            }
            return result;
        }
    }

    private static class Dot2DTask extends RecursiveTask<float[][]> {
        private static final int THRESHOLD = 100; // Threshold for task splitting
        private final float[][] A;
        private final float[][] B;
        private final int start;
        private final int end;

        public Dot2DTask(float[][] A, float[][] B, int start, int end) {
            this.A = A;
            this.B = B;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[][] compute() {
            if (end - start <= THRESHOLD) {
                float[][] C = new float[A.length][B[0].length];
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < B[0].length; j++) {
                        float sum = 0;
                        for (int k = 0; k < A[0].length; k++) {
                            sum += A[i][k] * B[k][j];
                        }
                        C[i][j] = sum;
                    }
                }
                return C;
            } else {
                int mid = (start + end) / 2;
                Dot2DTask leftTask = new Dot2DTask(A, B, start, mid);
                Dot2DTask rightTask = new Dot2DTask(A, B, mid, end);
                invokeAll(leftTask, rightTask);
                float[][] leftResult = leftTask.join();
                float[][] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[][] combineResults(float[][] leftResult, float[][] rightResult) {
            float[][] result = new float[A.length][B[0].length];
            for (int i = 0; i < A.length; i++) {
                for (int j = 0; j < B[0].length; j++) {
                    if (i < leftResult.length) {
                        result[i][j] = leftResult[i][j];
                    } else {
                        result[i][j] = rightResult[i - leftResult.length][j];
                    }
                }
            }
            return result;
        }
    }

    private static class TransposeTask extends RecursiveTask<float[][]> {
        private static final int THRESHOLD = 100;
        private final float[][] matrix;
        private final int start;
        private final int end;

        public TransposeTask(float[][] matrix, int start, int end) {
            this.matrix = matrix;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[][] compute() {
            if (end - start <= THRESHOLD) {
                int numRows = matrix.length;
                int numCols = matrix[0].length;
                float[][] transposed = new float[numCols][numRows];
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < numCols; j++) {
                        transposed[j][i] = matrix[i][j];
                    }
                }
                return transposed;
            } else {
                int mid = (start + end) / 2;
                TransposeTask leftTask = new TransposeTask(matrix, start, mid);
                TransposeTask rightTask = new TransposeTask(matrix, mid, end);
                invokeAll(leftTask, rightTask);
                float[][] leftResult = leftTask.join();
                float[][] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[][] combineResults(float[][] leftResult, float[][] rightResult) {
            float[][] result = new float[leftResult.length + rightResult.length][];
            System.arraycopy(leftResult, 0, result, 0, leftResult.length);
            System.arraycopy(rightResult, 0, result, leftResult.length, rightResult.length);
            return result;
        }
    }

    private static class OneHotEncodingTask extends RecursiveTask<float[][]> {
        private static final int THRESHOLD = 100;
        private final float[] Y;
        private final int numClasses;
        private final int start;
        private final int end;

        public OneHotEncodingTask(float[] Y, int numClasses, int start, int end) {
            this.Y = Y;
            this.numClasses = numClasses;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[][] compute() {
            if (end - start <= THRESHOLD) {
                float[][] oneHot = new float[numClasses][Y.length];
                for (int i = start; i < end; i++) {
                    int label = (int) Y[i];
                    oneHot[label][i] = 1;
                }
                return oneHot;
            } else {
                int mid = (start + end) / 2;
                OneHotEncodingTask leftTask = new OneHotEncodingTask(Y, numClasses, start, mid);
                OneHotEncodingTask rightTask = new OneHotEncodingTask(Y, numClasses, mid, end);
                invokeAll(leftTask, rightTask);
                float[][] leftResult = leftTask.join();
                float[][] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[][] combineResults(float[][] leftResult, float[][] rightResult) {
            float[][] result = new float[numClasses][leftResult[0].length];
            for (int i = 0; i < leftResult.length; i++) {
                System.arraycopy(leftResult[i], 0, result[i], 0, leftResult[i].length);
                System.arraycopy(rightResult[i], 0, result[i], leftResult[i].length, rightResult[i].length);
            }
            return result;
        }
    }

    private static class Subtract1DTask extends RecursiveTask<float[]> {
        private static final int THRESHOLD = 100;
        private final float[] A;
        private final float[] B;
        private final int start;
        private final int end;

        public Subtract1DTask(float[] A, float[] B, int start, int end) {
            this.A = A;
            this.B = B;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[] compute() {
            if (end - start <= THRESHOLD) {
                float[] C = new float[A.length];
                for (int i = start; i < end; i++) {
                    C[i] = A[i] - B[i];
                }
                return C;
            } else {
                int mid = (start + end) / 2;
                Subtract1DTask leftTask = new Subtract1DTask(A, B, start, mid);
                Subtract1DTask rightTask = new Subtract1DTask(A, B, mid, end);
                invokeAll(leftTask, rightTask);
                float[] leftResult = leftTask.join();
                float[] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[] combineResults(float[] leftResult, float[] rightResult) {
            float[] result = new float[leftResult.length + rightResult.length];
            System.arraycopy(leftResult, 0, result, 0, leftResult.length);
            System.arraycopy(rightResult, 0, result, leftResult.length, rightResult.length);
            return result;
        }
    }

    private static class Subtract2DTask extends RecursiveTask<float[][]> {
        private static final int THRESHOLD = 100;
        private final float[][] A;
        private final float[][] B;
        private final int start;
        private final int end;

        public Subtract2DTask(float[][] A, float[][] B, int start, int end) {
            this.A = A;
            this.B = B;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[][] compute() {
            if (end - start <= THRESHOLD) {
                float[][] C = new float[A.length][A[0].length];
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < A[0].length; j++) {
                        C[i][j] = A[i][j] - B[i][j];
                    }
                }
                return C;
            } else {
                int mid = (start + end) / 2;
                Subtract2DTask leftTask = new Subtract2DTask(A, B, start, mid);
                Subtract2DTask rightTask = new Subtract2DTask(A, B, mid, end);
                invokeAll(leftTask, rightTask);
                float[][] leftResult = leftTask.join();
                float[][] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[][] combineResults(float[][] leftResult, float[][] rightResult) {
            float[][] result = new float[leftResult.length + rightResult.length][];
            System.arraycopy(leftResult, 0, result, 0, leftResult.length);
            System.arraycopy(rightResult, 0, result, leftResult.length, rightResult.length);
            return result;
        }
    }

    private static class MultiplyScalar2DTask extends RecursiveTask<float[][]> {
        private static final int THRESHOLD = 100;
        private final float[][] A;
        private final float scalar;
        private final int start;
        private final int end;

        public MultiplyScalar2DTask(float[][] A, float scalar, int start, int end) {
            this.A = A;
            this.scalar = scalar;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[][] compute() {
            if (end - start <= THRESHOLD) {
                float[][] C = new float[A.length][A[0].length];
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < A[0].length; j++) {
                        C[i][j] = A[i][j] * scalar;
                    }
                }
                return C;
            } else {
                int mid = (start + end) / 2;
                MultiplyScalar2DTask leftTask = new MultiplyScalar2DTask(A, scalar, start, mid);
                MultiplyScalar2DTask rightTask = new MultiplyScalar2DTask(A, scalar, mid, end);
                invokeAll(leftTask, rightTask);
                float[][] leftResult = leftTask.join();
                float[][] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[][] combineResults(float[][] leftResult, float[][] rightResult) {
            float[][] result = new float[leftResult.length + rightResult.length][];
            System.arraycopy(leftResult, 0, result, 0, leftResult.length);
            System.arraycopy(rightResult, 0, result, leftResult.length, rightResult.length);
            return result;
        }
    }

    private static class MultiplyScalar1DTask extends RecursiveTask<float[]> {
        private static final int THRESHOLD = 100;
        private final float[] A;
        private final float scalar;
        private final int start;
        private final int end;

        public MultiplyScalar1DTask(float[] A, float scalar, int start, int end) {
            this.A = A;
            this.scalar = scalar;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[] compute() {
            if (end - start <= THRESHOLD) {
                float[] C = new float[A.length];
                for (int i = start; i < end; i++) {
                    C[i] = A[i] * scalar;
                }
                return C;
            } else {
                int mid = (start + end) / 2;
                MultiplyScalar1DTask leftTask = new MultiplyScalar1DTask(A, scalar, start, mid);
                MultiplyScalar1DTask rightTask = new MultiplyScalar1DTask(A, scalar, mid, end);
                invokeAll(leftTask, rightTask);
                float[] leftResult = leftTask.join();
                float[] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[] combineResults(float[] leftResult, float[] rightResult) {
            float[] result = new float[leftResult.length + rightResult.length];
            System.arraycopy(leftResult, 0, result, 0, leftResult.length);
            System.arraycopy(rightResult, 0, result, leftResult.length, rightResult.length);
            return result;
        }
    }

    private static class SumTask extends RecursiveTask<float[]> {
        private static final int THRESHOLD = 100;
        private final float[][] A;
        private final int start;
        private final int end;

        public SumTask(float[][] A, int start, int end) {
            this.A = A;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[] compute() {
            if (end - start <= THRESHOLD) {
                float[] C = new float[A[0].length];
                for (int j = start; j < end; j++) {
                    float sum = 0;
                    for (int i = 0; i < A.length; i++) {
                        sum += A[i][j];
                    }
                    C[j] = sum;
                }
                return C;
            } else {
                int mid = (start + end) / 2;
                SumTask leftTask = new SumTask(A, start, mid);
                SumTask rightTask = new SumTask(A, mid, end);
                invokeAll(leftTask, rightTask);
                float[] leftResult = leftTask.join();
                float[] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[] combineResults(float[] leftResult, float[] rightResult) {
            float[] result = new float[leftResult.length + rightResult.length];
            System.arraycopy(leftResult, 0, result, 0, leftResult.length);
            System.arraycopy(rightResult, 0, result, leftResult.length, rightResult.length);
            return result;
        }
    }

    private static class DerivateReLUTask extends RecursiveTask<float[][]> {
        private static final int THRESHOLD = 100;
        private final float[][] inputs;
        private final int start;
        private final int end;

        public DerivateReLUTask(float[][] inputs, int start, int end) {
            this.inputs = inputs;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[][] compute() {
            if (end - start <= THRESHOLD) {
                float[][] outputs = new float[inputs.length][inputs[0].length];
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < inputs[0].length; j++) {
                        outputs[i][j] = inputs[i][j] > 0 ? 1 : 0;
                    }
                }
                return outputs;
            } else {
                int mid = (start + end) / 2;
                DerivateReLUTask leftTask = new DerivateReLUTask(inputs, start, mid);
                DerivateReLUTask rightTask = new DerivateReLUTask(inputs, mid, end);
                invokeAll(leftTask, rightTask);
                float[][] leftResult = leftTask.join();
                float[][] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[][] combineResults(float[][] leftResult, float[][] rightResult) {
            float[][] result = new float[leftResult.length + rightResult.length][];
            System.arraycopy(leftResult, 0, result, 0, leftResult.length);
            System.arraycopy(rightResult, 0, result, leftResult.length, rightResult.length);
            return result;
        }
    }

    private static class Multiply2DTask extends RecursiveTask<float[][]> {
        private static final int THRESHOLD = 100;
        private final float[][] A;
        private final float[][] B;
        private final int start;
        private final int end;

        public Multiply2DTask(float[][] A, float[][] B, int start, int end) {
            this.A = A;
            this.B = B;
            this.start = start;
            this.end = end;
        }

        @Override
        protected float[][] compute() {
            if (end - start <= THRESHOLD) {
                float[][] C = new float[A.length][A[0].length];
                for (int i = start; i < end; i++) {
                    for (int j = 0; j < A[0].length; j++) {
                        C[i][j] = A[i][j] * B[i][j];
                    }
                }
                return C;
            } else {
                int mid = (start + end) / 2;
                Multiply2DTask leftTask = new Multiply2DTask(A, B, start, mid);
                Multiply2DTask rightTask = new Multiply2DTask(A, B, mid, end);
                invokeAll(leftTask, rightTask);
                float[][] leftResult = leftTask.join();
                float[][] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private float[][] combineResults(float[][] leftResult, float[][] rightResult) {
            float[][] result = new float[leftResult.length + rightResult.length][];
            System.arraycopy(leftResult, 0, result, 0, leftResult.length);
            System.arraycopy(rightResult, 0, result, leftResult.length, rightResult.length);
            return result;
        }
    }

    private static class ArgmaxTask extends RecursiveTask<int[]> {
        private static final int THRESHOLD = 100;
        private final float[][] A;
        private final int start;
        private final int end;

        public ArgmaxTask(float[][] A, int start, int end) {
            this.A = A;
            this.start = start;
            this.end = end;
        }

        @Override
        protected int[] compute() {
            if (end - start <= THRESHOLD) {
                int[] maxIndices = new int[A[0].length];
                for (int j = start; j < end; j++) {
                    float max = A[0][j];
                    int maxIndex = 0;
                    for (int i = 1; i < A.length; i++) {
                        if (A[i][j] > max) {
                            max = A[i][j];
                            maxIndex = i;
                        }
                    }
                    maxIndices[j] = maxIndex;
                }
                return maxIndices;
            } else {
                int mid = (start + end) / 2;
                ArgmaxTask leftTask = new ArgmaxTask(A, start, mid);
                ArgmaxTask rightTask = new ArgmaxTask(A, mid, end);
                invokeAll(leftTask, rightTask);
                int[] leftResult = leftTask.join();
                int[] rightResult = rightTask.join();
                return combineResults(leftResult, rightResult);
            }
        }

        private int[] combineResults(int[] leftResult, int[] rightResult) {
            int[] result = new int[leftResult.length + rightResult.length];
            System.arraycopy(leftResult, 0, result, 0, leftResult.length);
            System.arraycopy(rightResult, 0, result, leftResult.length, rightResult.length);
            return result;
        }
    }

}
