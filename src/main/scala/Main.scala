package org.ionkin.ml.test

import smile.classification
import smile.classification.{Classifier, KNN, OneVersusOne}
import smile.data.DataFrame
import smile.math.MathEx
import smile.math.distance.{EuclideanDistance, Metric}
import smile.math.kernel.GaussianKernel
import smile.validation.ClassificationValidations
import smile.validation.metric.Error
import smile.read

import java.nio.file.Paths
import java.util.Random
import java.util.function.BiFunction
import scala.reflect.ClassTag
import scala.util.Try

object Main {

  def extract_x_normalized_y(data: DataFrame): (Array[Array[Double]], Array[Int]) = {
    val y: Array[Int] = data.intVector(0).toIntArray.map(x => x - 1)
    val x: Array[Array[Double]] = data.drop(0).toArray()
    MathEx.normalize(x)
    (x, y)
  }

  def get_test_indexes(testSize: Int): Set[Int] =
    new Random(42).ints(testSize, 0, testSize).toArray.toSet

  def split_train_test[T : ClassTag](ar: Array[T]): (Array[T], Array[T]) = {
    val testSize = (ar.length * 0.3).toInt
    val testIndexes: Set[Int] = get_test_indexes(testSize)
    ar.zipWithIndex.partition { case (el, i) => !testIndexes.contains(i) } match {
      case (tr, ts) => (tr.map(_._1), ts.map(_._1))
    }
  }

  def my_knn(data: DataFrame, k: Int): ClassificationValidations[KNN[Array[Double]]] = {
    val (x, y) = extract_x_normalized_y(data)
    smile.validation.cv.classification(k = 10, x, y) { case (x, y) => classification.knn(x, y, k = k) }
  }

  def my_best_knn(data: DataFrame): (Int, ClassificationValidations[KNN[Array[Double]]]) =
    (1 until 30).map(k => (k, my_knn(data, k))).maxBy(_._2.avg.accuracy)

  def my_svm(data: DataFrame, sigma: Double, regulation: Double): Double = {
    val (x, y) = extract_x_normalized_y(data)
    val (x_train, x_test) = split_train_test(x)
    val (y_train, y_test) = split_train_test(y)
    val kernel = new GaussianKernel(sigma)
    val trainer: BiFunction[Array[Array[Double]], Array[Int], Classifier[Array[Double]]] = { case (x, y) =>
      classification.svm(x, y, kernel, regulation)
    }
    val model = OneVersusOne.fit(x_train, y_train, trainer)
    val prediction = model.predict(x_test)
    val err = Error.of(y_test, prediction)
    1 - err.toDouble / y_test.length
  }

  def my_best_svm(data: DataFrame): (Double, Double, Double) = {
    val svm_accuracies = for {
      sigma <- (1 until 20).map(x => x.toDouble / 2)
      regulation <- (1 until 20).map(x => x.toDouble / 2)
    } yield (sigma, regulation, my_svm(data, sigma, regulation))
    svm_accuracies.maxBy(_._3)
  }

  def parzen_window(data: DataFrame, k: Int, step: Double): ClassificationValidations[KNN[Array[Double]]] = {
    val (x, y) = extract_x_normalized_y(data)
    val weightedDistance = new Metric[Array[Double]] {
      def core(el: Double): Double = if (math.abs(el) < 1) 1 - el * el else 0

      def d(x1: Array[Double], x2: Array[Double]): Double = {
        val weight = new Array[Double](x1.length)
        for (i <- weight.indices) {
          weight(i) = core((x1(i) - x2(i)) / step)
        }
        val distance = new EuclideanDistance(weight)
        distance.d(x1, x2)
      }
    }
    smile.validation.cv.classification(k = 10, x, y) { case (x, y) => classification.knn(x, y, k = k, weightedDistance) }
  }

  def my_best_parzen_window(data: DataFrame): (Int, Double, ClassificationValidations[KNN[Array[Double]]]) = {
    val ks = (1 until 30).toArray
    val steps = (1 until 40).toArray.map(_ * 0.1)
    (for (k <- ks; s <- steps; m <- Try(parzen_window(data, k, s)).toOption.toList)
      yield (k, s, m)
      ).maxBy(_._3.avg.accuracy)
  }

  def main(args: Array[String]): Unit = {
    val wineCsv = Paths.get(getClass.getClassLoader.getResource("wine.data").toURI).toFile
    val wine: DataFrame = read.csv(wineCsv.getAbsolutePath)
    println("best k-NN: " + my_best_knn(wine)) // k=3, accuracy=96.81% ± 3.44
    println("best SVM: " + my_best_svm(wine)) // sigma=1.5, regulation=6.0, accuracy=1.0
    println("best Parzen window: " + my_best_parzen_window(wine)) // k=3, step=0.8, accuracy=97.65% ± 4.11
  }
}