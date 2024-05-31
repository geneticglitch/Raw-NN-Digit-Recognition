package org.example;

import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.IOException;

public class savingModel {

    public static void saveWeightsAndBiases(LayerDense layer1, LayerDense layer2) {
        saveToFile(layer1.weights, "Layer1_Weights.txt");
        saveToFile(layer1.biases, "Layer1_Biases.txt");
        saveToFile(layer2.weights, "Layer2_Weights.txt");
        saveToFile(layer2.biases, "Layer2_Biases.txt");
    }

    private static void saveToFile(float[][] data, String fileName) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            for (float[] row : data) {
                for (float value : row) {
                    writer.write(value + " ");
                }
                writer.newLine();
            }
            System.out.println("Data saved to " + fileName);
        } catch (IOException e) {
            System.err.println("Error occurred while saving data to " + fileName + ": " + e.getMessage());
        }
    }

    private static void saveToFile(float[] data, String fileName) {
        try (BufferedWriter writer = new BufferedWriter(new FileWriter(fileName))) {
            for (float value : data) {
                writer.write(value + " ");
            }
            System.out.println("Data saved to " + fileName);
        } catch (IOException e) {
            System.err.println("Error occurred while saving data to " + fileName + ": " + e.getMessage());
        }
    }
}
