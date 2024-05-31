package org.example;

public class DataSet {
    private final float[][] features;
    private final float[] labels;

    public DataSet(float[][] features, float[] labels) {
        this.features = features;
        this.labels = labels;
    }

    public float[][] getFeatures() {
        return features;
    }

    public float[] getLabels() {
        return labels;
    }
}
