
public abstract class EmbeddingMethod {
  protected Dataset highDataset;
  protected Dataset lowDataset;
  protected Visualizer visualizer;

  public void setDataset(Dataset dataset) {
    this.highDataset = dataset;
  }

  public void setVisualizer(Visualizer visualizer) {
    this.visualizer = visualizer;
  }

  public void notifyVisualizer() {
    this.visualizer.update(this.lowDataset);
  }

  abstract public void init();
  abstract public void update();
}
