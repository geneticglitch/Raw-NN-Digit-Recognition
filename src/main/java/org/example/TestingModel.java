package org.example;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;
import java.util.List;

public class TestingModel {

    public static void main(String[] args) throws IOException {
        // Load weights and biases
        LayerDense layer1 = loadLayer("Layer1_Weights.txt", "Layer1_Biases.txt");
        LayerDense layer2 = loadLayer("Layer2_Weights.txt", "Layer2_Biases.txt");

        DataSet dataSet = Utils.readCSV("test.csv");
        float[][] X = dataSet.getFeatures();
        float[] Y = dataSet.getLabels();


        layer1.forward(X);
        float[][] Z1 = layer1.getOutputs();
        float[][] A1 = Utils.ReLu(Z1);

        layer2.forward(A1);
        float[][] Z2 = layer2.getOutputs();
        float[][] A2 = Utils.softmax(Z2);

        int correct = 0;

        for(int index = 0 ;index<Y.length;index++){
            double max = Integer.MIN_VALUE;
            int index_prec = 0;
            for(int i = 0; i<10;i++){
                if(A2[i][index]>max){
                    max = A2[i][index];
                    index_prec = i;
                }
            }
            System.out.println("Predicted: " + index_prec);
            System.out.println("Actual: " + (int) Y[index]);

            if(index_prec == (int) Y[index]){
                correct++;
            }


        }
        System.out.println("Total Test cases: " + Y.length);
        System.out.println("Total correct: " + correct);
        System.out.println("Accuracy: " + (correct*100.0/Y.length) + "%");



    }

    private static LayerDense loadLayer(String weightsFile, String biasesFile) {
        float[][] weights = loadMatrixFromFile(weightsFile);
        float[] biases = loadArrayFromFile(biasesFile);
        LayerDense layer = new LayerDense(weights[0].length, weights.length);
        layer.weights = weights;
        layer.biases = biases;
        return layer;
    }

    private static float[][] loadMatrixFromFile(String fileName) {
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            List<float[]> rows = new ArrayList<>();
            String line;
            while ((line = reader.readLine()) != null) {
                String[] values = line.split(" ");
                float[] row = new float[values.length];
                for (int i = 0; i < values.length; i++) {
                    row[i] = Float.parseFloat(values[i]);
                }
                rows.add(row);
            }
            return rows.toArray(new float[0][]);
        } catch (IOException e) {
            System.err.println("Error loading matrix from file: " + e.getMessage());
            return null;
        }
    }

    private static float[] loadArrayFromFile(String fileName) {
        try (BufferedReader reader = new BufferedReader(new FileReader(fileName))) {
            String line = reader.readLine();
            String[] values = line.split(" ");
            float[] array = new float[values.length];
            for (int i = 0; i < values.length; i++) {
                array[i] = Float.parseFloat(values[i]);
            }
            return array;
        } catch (IOException e) {
            System.err.println("Error loading array from file: " + e.getMessage());
            return null;
        }
    }

}
