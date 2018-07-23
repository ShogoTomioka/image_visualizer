
import java.lang.Math;
import java.util.Arrays;

public class MDS extends EmbeddingMethod {
  protected boolean isInitialized;
  protected double learningRate;
  protected double[][] highDistanceMatrix;
  protected double[][] lowDistanceMatrix;

  MDS() {
    this.isInitialized = false;
    this.learningRate = 0.003;
  }

  protected void calculateDistanceMatrix(Dataset dataset, double[][] distanceMatrix) {
    for (int i = 0; i < dataset.getNumSamples(); i++) {
      for (int j = i; j < dataset.getNumSamples(); j++) {
        double distance = 0.0;
        for (int k = 0; k < dataset.getNumDimensions(); k++) {
          double diff = dataset.getFeature(i, k) - dataset.getFeature(j, k);
          distance += diff * diff;
        }
        distance = Math.sqrt(distance);
        distanceMatrix[i][j] = distance;
        distanceMatrix[j][i] = distance;
      }
    }
  }

  @Override
  public void init() {
    this.isInitialized = false;
    if (this.highDataset.isEmpty() == false) {
      //
      int numSamples = this.highDataset.getNumSamples();
      //
      double[][] lowFeatures = new double[numSamples][2];
      String[] lowLabels = new String[numSamples];
      for (int i = 0; i < numSamples; i++) {
        lowFeatures[i][0] = Math.random();
        lowFeatures[i][1] = Math.random();
        lowLabels[i] = this.highDataset.getLabel(i);
      }
      this.lowDataset = new Dataset(lowLabels, lowFeatures);
      //
      this.highDistanceMatrix = new double[numSamples][numSamples];
      this.lowDistanceMatrix = new double[numSamples][numSamples];
      this.calculateDistanceMatrix(this.highDataset, this.highDistanceMatrix);
      //
      this.isInitialized = true;
    }
  }

  @Override
  public void update() {
    if (this.isInitialized == true) {
      // 低次元空間の距離行列を計算する．
      this.calculateDistanceMatrix(this.lowDataset, this.lowDistanceMatrix);
      // 勾配を計算する．
      int numSamples = this.highDataset.getNumSamples();
      double[][] gradients = new double[numSamples][2];
      for (int i = 0; i < numSamples; i++) {
        Arrays.fill(gradients[i], 0.0); // 0で初期化する．
        for (int j = 0; j < numSamples; j++) {
          double distanceError = (this.highDistanceMatrix[i][j] - this.lowDistanceMatrix[i][j]) /
                 (this.lowDistanceMatrix[i][j] + 1.0e-8);
          gradients[i][0] += distanceError * (this.lowDataset.getFeature(j, 0) - this.lowDataset.getFeature(i, 0));
          gradients[i][1] += distanceError * (this.lowDataset.getFeature(j, 1) - this.lowDataset.getFeature(i, 1));
        }
      }
      // 低次元空間の座標を更新する．o0
      for (int i = 0; i < numSamples; i++) {
        double newFeatureX = this.lowDataset.getFeature(i, 0) - this.learningRate * gradients[i][0];
        double newFeatureY = this.lowDataset.getFeature(i, 1) - this.learningRate * gradients[i][1];
        this.lowDataset.updateFeature(i, 0, newFeatureX);
        this.lowDataset.updateFeature(i, 1, newFeatureY);
      }
      // ビジュアライザーに低次元空間の座標の更新を伝える．
      this.notifyVisualizer();
    }
  }
}
