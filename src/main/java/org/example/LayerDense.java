package org.example;

public class LayerDense {
    float[][] weights;
    float[] biases;
    private float[][] outputs;

    public LayerDense(int nInputs, int nNeurons) {
        this.weights = new float[nNeurons][nInputs];
        this.biases = new float[nNeurons];
        // Initialize weights and biases betwwen -0.5 and 0.5
        for (int i = 0; i < nNeurons; i++) {
            for (int j = 0; j < nInputs; j++) {
                this.weights[i][j] = (float) (Math.random() - 0.5);
            }
            this.biases[i] = (float) (Math.random() - 0.5);
        }
    }

    public void forward(float[][] inputs) {
        this.outputs = new float[weights.length][inputs[0].length];
        for (int i = 0; i < weights.length; i++) {
            for (int j = 0; j < inputs[0].length; j++) {
                float sum = 0;
                for (int k = 0; k < inputs.length; k++) {
                    sum += weights[i][k] * inputs[k][j];
                }
                this.outputs[i][j] = sum + biases[i];
            }
        }
    }

    public float[][] getOutputs() {
        return outputs;
    }
}
