package org.example;

import java.io.IOException;

import java.io.FileOutputStream;
import java.io.ObjectOutputStream;
import org.example.savingModel;
import org.example.LayerDense;
import org.example.Utils;

public class Main {
    public static void main(String[] args) throws IOException {
        DataSet dataSet = Utils.readCSV("train.csv");
        float[][] X = dataSet.getFeatures();
        float[] Y = dataSet.getLabels();



        System.out.println("reading files complete");
        long startTime = System.currentTimeMillis();
        System.out.println(X.length);
        System.out.println(X[0].length);

        LayerDense layer1 = new LayerDense(X.length, 20);
        LayerDense layer2 = new LayerDense(20, 10);

        // Training parameters
        int numCycles = 100; // Total number of training cycles
        float learningRate = 0.008F; // Learning rate

        for (int cycle = 1; cycle <= numCycles; cycle++) {
            // Forward pass
            layer1.forward(X);
            float[][] Z1 = layer1.getOutputs();
            float[][] A1 = Utils.ReLu(Z1);

            layer2.forward(A1);
            float[][] Z2 = layer2.getOutputs();
            float[][] A2 = Utils.softmax(Z2);

            // Backward pass
            float[][] Y_one_hot = Utils.one_hot_encoding(Y, 10);
            float[][] dZ2 = Utils.subtract_2d(A2, Y_one_hot);
            float[][] dW2 = Utils.multiply_scalar_2d(Utils.dot_2d(dZ2, Utils.transpose(A1)), (float) (1.0 / X.length));
            float[] db2 = Utils.multiply_scalar_1d(Utils.sum(dZ2), (float) (1.0 / X.length));
            float[][] dZ1 = Utils.multiply_2d(Utils.dot_2d(Utils.transpose(layer2.weights), dZ2), Utils.derivate_ReLu(Z1));
            float[][] dW1 = Utils.multiply_scalar_2d(Utils.dot_2d(dZ1, Utils.transpose(X)), (float) (1.0 / X.length));
            float[] db1 = Utils.multiply_scalar_1d(Utils.sum(dZ1), (float) (1.0 / X.length));

            layer1.weights = Utils.subtract_2d(layer1.weights, Utils.multiply_scalar_2d(dW1, learningRate));
            layer1.biases = Utils.subtract_1d(layer1.biases, Utils.multiply_scalar_1d(db1, learningRate));
            layer2.weights = Utils.subtract_2d(layer2.weights, Utils.multiply_scalar_2d(dW2, learningRate));
            layer2.biases = Utils.subtract_1d(layer2.biases, Utils.multiply_scalar_1d(db2, learningRate));

            if (cycle % 2 == 0) {
                float accuracy = calculateAccuracy(X, Y, layer1, layer2);
                System.out.println("Cycle " + cycle + " - Accuracy: " + accuracy);
            }
        }

        savingModel.saveWeightsAndBiases(layer1, layer2);
        long endTime = System.currentTimeMillis();
        System.out.println("Training complete. Time elapsed: " + (endTime - startTime) / 1000 + " seconds");
    }

    public static float calculateAccuracy(float[][] X, float[] Y, LayerDense layer1, LayerDense layer2) {
        layer1.forward(X);
        float[][] Z1 = layer1.getOutputs();
        float[][] A1 = Utils.ReLu(Z1);

        layer2.forward(A1);
        float[][] Z2 = layer2.getOutputs();
        float[][] A2 = Utils.softmax(Z2);

        int[] predictions = Utils.argmax(A2);
        float correct = 0;
        for (int i = 0; i < Y.length; i++) {
            if (predictions[i] == Y[i]) {
                correct++;
            }
        }

        return correct / Y.length;
    }

}
