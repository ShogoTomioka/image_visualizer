
import java.lang.Math;
import java.util.Arrays;

public class SNE extends MDS {
  protected double[][] highTransitionProbabilityMatrix;
  protected double[][] lowTransitionProbabilityMatrix;

  SNE() {
    this.isInitialized = false;
    this.learningRate = 0.1;
  }

  protected void calculateProbabilityMatrix(double[][] distanceMatrix, double[][] probabilityMatrix, double sigma) {
    int numSamples = distanceMatrix.length;
    for (int i = 0; i < numSamples; i++) {
      for (int j = i + 1; j < numSamples; j++) {
        double similarity = Math.exp(-(distanceMatrix[i][j] * distanceMatrix[i][j]) / sigma);
        probabilityMatrix[i][j] = similarity;
        probabilityMatrix[j][i] = similarity;
      }
      probabilityMatrix[i][i] = 0.0;
    }
    for (int i = 0; i < numSamples; i++) {
      double degree = 0.0;
      for (int j = 0; j < numSamples; j++) {
        degree += probabilityMatrix[i][j];
      }
      for (int j = 0; j < numSamples; j++) {
        probabilityMatrix[i][j] /= degree;
      }
    }
  }

  @Override
  public void init() {
    this.isInitialized = false;
    if (this.highDataset.isEmpty() == false) {
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
      this.highTransitionProbabilityMatrix = new double[numSamples][numSamples];
      this.lowTransitionProbabilityMatrix = new double[numSamples][numSamples];
      this.calculateDistanceMatrix(this.highDataset, this.highDistanceMatrix);
      this.calculateProbabilityMatrix(this.highDistanceMatrix, this.highTransitionProbabilityMatrix, 0.08);
      this.isInitialized = true;
    }
  }

  @Override
  public void update() {
    if (this.isInitialized == true) {
      //
      int numSamples = this.highDataset.getNumSamples();
      // 低次元空間の距離行列と遷移確率行列を計算する．
      this.calculateDistanceMatrix(this.lowDataset, this.lowDistanceMatrix);
      this.calculateProbabilityMatrix(this.lowDistanceMatrix, this.lowTransitionProbabilityMatrix, 1.0);
      // 勾配を計算する．
      double[][] gradients = new double[numSamples][2];
      for (int i = 0; i < numSamples; i++) {
        Arrays.fill(gradients[i], 0.0);
        for (int j = 0; j < numSamples; j++) {
          double probability_error = (this.highTransitionProbabilityMatrix[i][j] - this.lowTransitionProbabilityMatrix[i][j] +
            this.highTransitionProbabilityMatrix[j][i] - this.lowTransitionProbabilityMatrix[j][i]);
          gradients[i][0] += probability_error * (this.lowDataset.getFeature(i, 0) - this.lowDataset.getFeature(j, 0));
          gradients[i][1] += probability_error * (this.lowDataset.getFeature(i, 1) - this.lowDataset.getFeature(j, 1));
        }
      }
      // 低次元空間の座標を更新する．
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
