
import java.lang.Math;
import java.util.Arrays;

public class NLM extends MDS {
  NLM() {
    this.isInitialized = false;
    this.learningRate = 0.01;
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
                 (this.highDistanceMatrix[i][j] * this.lowDistanceMatrix[i][j] + 1.0e-8);
          gradients[i][0] += distanceError * (this.lowDataset.getFeature(j, 0) - this.lowDataset.getFeature(i, 0));
          gradients[i][1] += distanceError * (this.lowDataset.getFeature(j, 1) - this.lowDataset.getFeature(i, 1));
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
