package org.ionkin.ml.test

import com.typesafe.scalalogging.StrictLogging
import smile.validation.metric.MSE

import java.io.File
import java.nio.file.Paths

object Main extends StrictLogging {

  def wine(): Unit = {
    val winePath = Paths.get(getClass.getClassLoader.getResource("wine.csv").toURI).toFile
    ClassificationAlgorithms.show_best(winePath, y_column_id = 0)
    // accuracy=99.4% for RDA for alpha=0.1:0.1:1 (maybe with few exceptions)
    // MLP accuracy=0.9339869281045754 for epochs=100, layers=ReLU(41), mini_batch=1
    // k=3, accuracy=96.81% ± 3.44 for k-NN
    // sigma=1.5, regulation=6.0, accuracy=1.0 (100%)  for SVM
    // k=3, h=0.8, accuracy=97.65% ± 4.11 for Parzen window
  }

  def spam(): Unit = {
    val spamPath = Paths.get(getClass.getClassLoader.getResource("spambase.csv").toURI).toFile
    ClassificationAlgorithms.show_best(spamPath, y_column_id = -1)
    // RDA: accuracy=89.65% for alpha=0.1. very fast optimization
    // MLP accuracy=0.9200178806700545 for epochs=100, layers=ReLU(21), mini_batch=1,
    // optimization for MLP can be really slow (1-30 seconds per iteration)
    // k=1, accuracy=91.33% ± 1.29 for k-NN
    // sigma=0.25, regulation=9.5, epochs=2, accuracy=0.838 (83.8%) for SVM
    // k=1, h=1.2, accuracy=91.68% ± 1.05 for Parzen window
  }

  /** @see {@link smile.feature.imputation} to handle missing values */
  def cancer(): Unit = {
    val spamPath = Paths.get(getClass.getClassLoader.getResource("breast-cancer-wisconsin.csv").toURI).toFile
    ClassificationAlgorithms.show_best(spamPath, y_column_id = -1)
    // best RDA: alpha=0.6, accuracy=0.9555771365149832
    // best k-NN: k=7, accuracy=96.83% ± 2.55
    // best Parzen window: k=5, h=2.4, accuracy=96.68% ± 2.39
    // best SVM: kernel=linear, C=0.5, epochs=2, accuracy=0.41
    // best MLP: epochs=100, layers=ReLU(41), mini_batch=1, accuracy=0.964
  }

  def servo(): Unit = {
    val spamPath = Paths.get(getClass.getClassLoader.getResource("servo.csv").toURI).toFile
    RegressionAlgorithms.show_best(spamPath, y_column_id = -1)
    // SVM mse=0.50 for epsilon=0.001, kernel=Gaussian(0.05), C=20
  }

  def ya_data(): Unit = {
    val homePath = "/home/mika"
    val spoiledPath = new File(s"$homePath/Data/YaData/spoiled_data.csv")
    val filled = RegressionAlgorithms.extract(spoiledPath, y_column_id = 0)._1
    val clean_file = new File(s"$homePath/Data/YaData/clean_data.csv")
    val clean_data =  RegressionAlgorithms.extract(clean_file, y_column_id = 0)._1
    var mse: Double = 0
    for (k <- filled.indices) {
      mse += MSE.of(clean_data(k), filled(k))
    }
    println(s"ya_data MSE: ${mse / filled.length}")
  }

  def main_classification(): Unit = {
    logger.info("start classification")
    wine()
    spam()
    cancer()
  }

  def main_regression(): Unit = {
    logger.info("start regression")
    servo()
  }

  def main(args: Array[String]): Unit = {
    main_classification()
    main_regression()
  }
}