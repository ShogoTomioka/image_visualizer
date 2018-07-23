
import java.awt.event.ActionListener;
import java.awt.event.ActionEvent;
import java.awt.BorderLayout;
import java.awt.Container;
import java.awt.Color;
import java.awt.Panel;


import java.io.File;

import javax.swing.JButton;
import javax.swing.JFileChooser;
import javax.swing.JFrame;
import javax.swing.JPanel;
import javax.swing.SwingUtilities;
import javax.swing.Timer;

public class JExperience {

  public static void main(String[] args) {
    SwingUtilities.invokeLater(new Runnable() {
      @Override
      public void run() {
        JExperience exp = new JExperience();
        exp.exec();
      }
    });
  }

  public void exec() {
    // ウィンドウを作成する．
    final JFrame frame = new JFrame("Java Experience");
    // 特徴ベクトルを読み込む．
    //Dataset dataset = CsvDatasetReader.readDataset("usps.csv");
    Dataset dataset = CsvDatasetReader.readDataset("mnist.csv");
    //Dataset dataset = CsvDatasetReader.readDataset("scurve.csv");
    //Dataset dataset = CsvDatasetReader.readDataset("glove.csv");

    // 描画画面を作成する．
    //final VPanel visualizer = new VPanel();
    final MnistPanel visualizer = new MnistPanel();

    // マッピングモデルを作成する．
    final EmbeddingMethod embMethod = new MDS();
    //final EmbeddingMethod embMethod = new NLM();
    //final EmbeddingMethod embMethod = new SNE();

    // マッピングモデルに特徴ベクトルをセットする．
    embMethod.setDataset(dataset);

    // マッピングモデルのオブサーバーとして描画画面を登録する．
    embMethod.setVisualizer(visualizer);

    // タイマーを作成する．
    final Timer timer = new Timer(50, new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        embMethod.update();
      }
    });

    // ボタンを追加する．
    final JPanel panelButton = new JPanel();
    //final JButton buttonRead = new JButton("Read");
    final JButton buttonRun = new JButton("Run");
    final JButton buttonStop = new JButton("Stop");
    //panelButton.add(buttonRead);
    panelButton.add(buttonRun);
    panelButton.add(buttonStop);

    // 匿名クラス（Anonymous Class）を用いてボタンのイベントを追加する．
    //buttonRead.addActionListener(new ActionListener() {
    //  @Override
    //  public void actionPerformed(ActionEvent e) {
      	//    // ファイル選択ダイアログを表示する．
    //    JFileChooser fileChooser = new JFileChooser();
    //    final int selected = fileChooser.showOpenDialog(frame);
    //    if (selected == JFileChooser.APPROVE_OPTION) {
    //      File file = fileChooser.getSelectedFile();
    //      Dataset dataset = CsvDatasetReader.readDataset(file.getName());
    //      embMethod.setDataset(dataset);
    //      timer.stop();
    //    }
    //  }
    //});

    buttonRun.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
		// パネルの背景を変更する
//      	panelButton.setBackground(Color.BLUE);
			visualizer.setBackground(Color.BLUE);
      	// 最適化スタート
        embMethod.init();
        timer.start();
      }
    });

    buttonStop.addActionListener(new ActionListener() {
      @Override
      public void actionPerformed(ActionEvent e) {
        // 最適化ストップ
        timer.stop();
      }
    });


    // ウィンドウのレイアウトを設定する．
    final Container content = frame.getContentPane();
    content.setLayout(new BorderLayout());

    // コンポーネントを追加する．
    content.add(visualizer, BorderLayout.PAGE_START);
    content.add(panelButton, BorderLayout.PAGE_END);

    // ウィンドウのサイズをコンテンツのサイズに合わせる．
    frame.pack();

    // ウィンドウの位置をスクリーンの中央に指定する．
    frame.setLocationRelativeTo(null);

    // ウィンドウの閉じるボタンが押されたらプログラムを終了する．
    frame.setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);

    // ウィンドウを表示する．
    frame.setResizable(false);
    frame.setVisible(true);
  }
}
