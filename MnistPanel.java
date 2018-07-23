
import java.lang.Math;
import java.util.Map;
import java.util.HashMap;
import javax.swing.JPanel;
import java.awt.Dimension;
import java.awt.Graphics;
import java.awt.Color;
import java.awt.image.BufferedImage;

import javax.imageio.ImageIO;
import java.io.File;

public class MnistPanel extends JPanel implements Visualizer {
  private int panelWidth;
  private int panelHeight;
  private BufferedImage buffer;
  private Color[] pallet;
  private BufferedImage[] mnist_imgs;

  MnistPanel() {
    // パネルの大きさを固定する．
    this.panelWidth = 640;
    this.panelHeight = 640;
    this.setPreferredSize(new Dimension(this.panelWidth, this.panelHeight));
    // パネルの大きさに合わせたバッファ領域を用意する．
    this.buffer = new BufferedImage(this.panelWidth, this.panelHeight, BufferedImage.TYPE_INT_RGB);
    Graphics bg = this.buffer.createGraphics();
    bg.setColor(new Color(236,239,241));
    bg.fillRect(0, 0, this.panelWidth, this.panelHeight);
    // ラベルごとに異なる色を割り当てるためのパレットを用意する．
    this.pallet = new Color[10];
    this.pallet[0] = new Color(0,138,0,128);
    this.pallet[1] = new Color(162,0,37,128);
    this.pallet[2] = new Color(96,169,23,128);
    this.pallet[3] = new Color(240,163,10,128);
    this.pallet[4] = new Color(0,80,239,128);
    this.pallet[5] = new Color(106,0,255,128);
    this.pallet[6] = new Color(227,200,0,128);
    this.pallet[7] = new Color(130,90,44,128);
    this.pallet[8] = new Color(109,135,100,128);
    this.pallet[9] = new Color(118,96,138,128);
    // USPSの画像を読み込む．
    int numImages = 500;
    this.mnist_imgs = new BufferedImage[numImages];
    for (int i = 0; i < numImages; i++) {
      try {
        String filename = "./mnist_imgs/mnist_" + (i + 1) + ".png";
        this.mnist_imgs[i] = ImageIO.read(new File(filename));
      } catch (Exception e) {
        e.printStackTrace();
        this.mnist_imgs[i] = null;
      }
    }
  }

  @Override
  public void update(Dataset dataset) {
    // ラベルに色を割り当てる．
    int numSamples = dataset.getNumSamples();
    Map<String,Color> colorMap = new HashMap<>();
    int counter = 0;
    for (int i = 0; i < numSamples; i++) {
      if (colorMap.containsKey(dataset.getLabel(i)) == false) {
        colorMap.put(dataset.getLabel(i), this.pallet[counter % this.pallet.length]);
        counter += 1;
      }
    }

    // 座標値をパネルの大きさに合わせるため，中心と最大半径を求める．
    double minX = 0.0;
    double maxX = 0.0;
    double minY = 0.0;
    double maxY = 0.0;
    for (int i = 0; i < numSamples; i++) {
      minX = Math.min(minX, dataset.getFeature(i, 0));
      maxX = Math.max(maxX, dataset.getFeature(i, 0));
      minY = Math.min(minY, dataset.getFeature(i, 1));
      maxY = Math.max(maxY, dataset.getFeature(i, 1));
    }
    double centerX = (maxX - Math.abs(minX)) / 2.0;
    double centerY = (maxY - Math.abs(minY)) / 2.0;
    double maxDistance = 0.0;
    for (int i = 0; i < numSamples; i++) {
      double diffX = dataset.getFeature(i, 0) - centerX;
      double diffY = dataset.getFeature(i, 1) - centerY;
      double distance = Math.sqrt(diffX * diffX + diffY * diffY);
      if (maxDistance < distance) {
        maxDistance = distance;
      }
    }

    // バッファに描画していく．
    Graphics bg = this.buffer.createGraphics();
    bg.setColor(new Color(255,255,255));
    bg.fillRect(0, 0, this.panelWidth, this.panelHeight);
    for (int i = 0; i < numSamples; i++) {
      // パネルに合わせた座標値を計算する．
      int x = (int)Math.floor(((dataset.getFeature(i, 0) - centerX) / maxDistance + 1.0) * (this.panelWidth / 2.0));
      int y = (int)(this.panelHeight - Math.floor(((dataset.getFeature(i, 1) - centerY) / maxDistance + 1.0) * (this.panelHeight / 2.0)));
      // ラベルに合わせた色を得る．
      bg.setColor(colorMap.get(dataset.getLabel(i)));
      bg.fillOval(x, y, 20, 20);
      // データ点とラベルを描画する．
      bg.drawImage(this.mnist_imgs[i], x - 1, y, this);
    }

    // 再描画する．
    repaint();
  }

  @Override
  public void paintComponent(Graphics g) {
    super.paintComponent(g);
    // バッファの画像をパネルにコピーする．
    g.drawImage(buffer, 0, 0, this);
  }
}
