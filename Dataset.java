
public class Dataset {
  private double[][] features;
  private String[] labels;
  private int numSamples;
  private int numDimensions;

  // コンストラクタ
  Dataset() {
    this.numSamples = -1;
    this.numDimensions = -1;
  }

  Dataset(String[] labels, double[][] features) {
    this.labels = labels;
    this.features = features;
    this.numSamples = features.length;
    this.numDimensions = features[0].length;
  }

  // 特徴ベクトルの値の更新
  public void updateFeature(int sid, int did, double value) {
    if (sid >= 0 && sid < numSamples && did >= 0 && did < numDimensions) {
      this.features[sid][did] = value;
    }
  }

  // ラベルの更新
  public void updateLabel(int sid, String label) {
    if (sid >= 0 && sid < numSamples) {
      this.labels[sid] = label;
    }
  }

  // 特徴ベクトルの値の取得
  public double getFeature(int sid, int did) {
    if (sid >= 0 && sid < numSamples && did >= 0 && did < numDimensions) {
      return this.features[sid][did];
    }
    return 0;
  }

  // ラベルの取得
  public String getLabel(int sid) {
    if (sid >= 0 && sid < numSamples) {
      return this.labels[sid];
    }
    return "";
  }

  // サンプル数の取得
  public int getNumSamples() {
    return this.numSamples;
  }

  // 次元数の取得
  public int getNumDimensions() {
    return this.numDimensions;
  }

  // 空っぽかどうかの判定
  public boolean isEmpty() {
    if (this.numSamples <=0 || this.numDimensions <= 0) {
       return true;
    }
    return false;
  }
}
