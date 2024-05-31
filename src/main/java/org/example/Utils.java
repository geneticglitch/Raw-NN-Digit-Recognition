package org.example;

import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;

import java.io.FileReader;
import java.io.IOException;
import java.io.Reader;
import java.util.ArrayList;
import java.util.List;

import java.util.concurrent.*;

public class Utils {
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
        float[][] outputs = new float[inputs.length][inputs[0].length];
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                outputs[i][j] = Math.max(0, inputs[i][j]);
            }
        }
        return outputs;
    }

    public static float[][] softmax(float[][] Z) {
        float[][] A = new float[Z.length][Z[0].length];

        float[][] expZ = new float[Z.length][Z[0].length];
        for (int i = 0; i < Z.length; i++) {
            for (int j = 0; j < Z[i].length; j++) {
                expZ[i][j] = (float) Math.exp(Z[i][j]);
            }
        }

        // Compute softmax row vise
        for (int j = 0; j < Z[0].length; j++) {
            float sumExpZ = 0.0F;
            for (int i = 0; i < Z.length; i++) {
                sumExpZ += expZ[i][j];
            }
            for (int i = 0; i < Z.length; i++) {
                A[i][j] = expZ[i][j] / sumExpZ;
            }
        }

        return A;
    }

    public static float[][] one_hot_encoding(float[] Y, int numClasses) {
        float[][] oneHot = new float[numClasses][Y.length];
        for (int i = 0; i < Y.length; i++) {
            int label = (int) Y[i];
            oneHot[label][i] = 1;
        }
        return oneHot;
    }

    public static float[] subtract_1d(float[] A, float[] B) {
        float[] C = new float[A.length];
        for (int i = 0; i < A.length; i++) {
            C[i] = A[i] - B[i];
        }
        return C;
    }

    public static float[][] subtract_2d(float[][] A, float[][] B) {
        float[][] C = new float[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                C[i][j] = A[i][j] - B[i][j];
            }
        }
        return C;
    }

    public static float[][] dot_2d(float[][] A, float[][] B) {
        float[][] C = new float[A.length][B[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < B[0].length; j++) {
                float sum = 0;
                for (int k = 0; k < A[0].length; k++) {
                    sum += A[i][k] * B[k][j];
                }
                C[i][j] = sum;
            }
        }
        return C;
    }

    public static float[][] transpose(float[][] matrix) {
        int numRows = matrix.length;
        int numCols = matrix[0].length;

        float[][] transposed = new float[numCols][numRows];

        for (int i = 0; i < numRows; i++) {
            for (int j = 0; j < numCols; j++) {
                transposed[j][i] = matrix[i][j];
            }
        }

        return transposed;
    }


   public static float[][] multiply_scalar_2d(float[][] A, float scalar) {
        float[][] C = new float[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                C[i][j] = A[i][j] * scalar;
            }
        }
        return C;
    }

    public static float[] multiply_scalar_1d(float[] A, float scalar) {
        float[] C = new float[A.length];
        for (int i = 0; i < A.length; i++) {
            C[i] = A[i] * scalar;
        }
        return C;
    }

    public static float[] sum(float[][] A) {
        float[] C = new float[A[0].length];
        for (int j = 0; j < A[0].length; j++) {
            float sum = 0;
            for (int i = 0; i < A.length; i++) {
                sum += A[i][j];
            }
            C[j] = sum;
        }
        return C;
    }

    public static float[][] derivate_ReLu(float[][] inputs) {
        float[][] outputs = new float[inputs.length][inputs[0].length];
        for (int i = 0; i < inputs.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                outputs[i][j] = inputs[i][j] > 0 ? 1 : 0;
            }
        }
        return outputs;
    }

    public static float[][] multiply_2d(float[][] A, float[][] B) {
        float[][] C = new float[A.length][A[0].length];
        for (int i = 0; i < A.length; i++) {
            for (int j = 0; j < A[0].length; j++) {
                C[i][j] = A[i][j] * B[i][j];
            }
        }
        return C;
    }

    public static int[] argmax(float[][] A) {
        int numRows = A.length;
        int numCols = A[0].length;

        int[] maxIndices = new int[numCols];

        for (int j = 0; j < numCols; j++) {
            float max = A[0][j];
            int maxIndex = 0;

            for (int i = 1; i < numRows; i++) {
                if (A[i][j] > max) {
                    max = A[i][j];
                    maxIndex = i;
                }
            }

            maxIndices[j] = maxIndex;
        }

        return maxIndices;
    }

}

