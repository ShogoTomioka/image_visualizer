
import java.nio.file.Files;
import java.nio.file.Paths;
import java.nio.charset.StandardCharsets;
import java.io.IOException;
import java.util.Arrays;
import java.lang.Math;

public class CsvDatasetReader {
  public static void main(String[] args) {
    Dataset dataset = readDataset("scurve.csv");
    System.out.println(dataset.getNumSamples());
    System.out.println(dataset.getNumDimensions());
    System.out.println(dataset.getLabel(0));
    System.out.println(dataset.getFeature(0,2));
    System.out.println(dataset.getLabel(499));
    System.out.println(dataset.getFeature(499,2));
  }

  public static Dataset readDataset(String filename) {
    try {
      // CSVファイルを読み込んで，Stringの配列に格納する．
      String[] lines = Files.readAllLines(Paths.get(filename), StandardCharsets.UTF_8).toArray(new String[0]);
      // サンプル数を得る．
      int numSamples = lines.length;
      // 特徴数を得る．1行目をカンマで分割し，生成される配列の大きさから1を引く（1列目はラベルのため）
      int numDimensions = lines[0].split(",").length - 1;
      //
      String[] labels = new String[numSamples];
      double[][] features = new double[numSamples][numDimensions];
      // データを読み込む．
      for (int i = 0; i < numSamples; i++) {
        // CSVのレコードをカンマで分割する．
        String[] values = lines[i].split(",");
        // ラベルをとりだす．
        labels[i] = values[0];
        // 特徴をとりだす（配列の範囲をはみ出さないようにする）．
        Arrays.fill(features[i], 0.0);
        for (int j = 0; j < Math.min(numDimensions, values.length - 1); j++) {
          features[i][j] = Double.parseDouble(values[1 + j]);
        }
      }
      return new Dataset(labels, features);
    } catch (IOException e) {
      e.printStackTrace();
      return new Dataset();
    }
  }
}
