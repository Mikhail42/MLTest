package org.ionkin.ml.test

import ClassificationAlgorithms.show_best

import com.typesafe.scalalogging.StrictLogging

import java.nio.file.Paths

object Main extends StrictLogging {

  def wine(): Unit = {
    val winePath = Paths.get(getClass.getClassLoader.getResource("wine.csv").toURI).toFile
    show_best(winePath, y_column_id = 0)
    // accuracy=99.4% for RDA for alpha=0.1:0.1:1 (maybe with few exceptions)
    // MLP accuracy=0.9339869281045754 for epochs=100, layers=ReLU(41), mini_batch=1
    // k=3, accuracy=96.81% ± 3.44 for k-NN
    // sigma=1.5, regulation=6.0, accuracy=1.0 (100%)  for SVM
    // k=3, h=0.8, accuracy=97.65% ± 4.11 for Parzen window
  }

  def spam(): Unit = {
    val spamPath = Paths.get(getClass.getClassLoader.getResource("spambase.csv").toURI).toFile
    show_best(spamPath, y_column_id = -1)
    // RDA: accuracy=89.65% for alpha=0.1. very fast optimization
    // MLP accuracy=0.9200178806700545 for epochs=100, layers=ReLU(21), mini_batch=1,
    // optimization for MLP can be really slow (1-30 seconds per iteration)
    // k=1, accuracy=91.33% ± 1.29 for k-NN
    // sigma=0.25, regulation=9.5, epochs=2, accuracy=0.838 (83.8%) for SVM
    // k=1, h=1.2, accuracy=91.68% ± 1.05 for Parzen window
  }

  def cancer(): Unit = {
    val spamPath = Paths.get(getClass.getClassLoader.getResource("breast-cancer-wisconsin.csv").toURI).toFile
    show_best(spamPath, y_column_id = -1)
    // best RDA: does not work with NaN
    // best k-NN: k=3, accuracy=95.86% ± 2.16
    // best Parzen window: k=3, h=2.8, accuracy=96.59% ± 2.11
    // best SVM: 0 accuracy (I have NaN values)
    // best MLP: accuracy < 70%
  }

  def main(args: Array[String]): Unit = {
    logger.info("start")
    wine()
    spam()
    cancer()
  }
}