package org.ionkin.ml.test

import com.typesafe.scalalogging.StrictLogging
import smile.classification.{Classifier, KNN, OneVersusOne, SVM}
import smile.data.DataFrame
import smile.hpo.Hyperparameters
import smile.math.MathEx
import smile.math.distance.{EuclideanDistance, Metric}
import smile.{classification, read}
import smile.validation.ClassificationValidations
import smile.validation.metric.Error

import java.nio.file.Paths
import java.util.function.BiFunction
import java.util.{Properties, Random}
import scala.reflect.ClassTag
import scala.util.Try

object Main extends StrictLogging {

  def extract_x_y(data: DataFrame): (Array[Array[Double]], Array[Int]) = {
    val y: Array[Int] = data.intVector(0).toIntArray.map(x => x - 1)
    val x: Array[Array[Double]] = data.drop(0).toArray()
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

  def my_knn(x: Array[Array[Double]], y: Array[Int], k: Int): ClassificationValidations[KNN[Array[Double]]] = {
    smile.validation.cv.classification(k = 10, x, y) { case (x, y) => classification.knn(x, y, k = k) }
  }

  def my_best_knn(x: Array[Array[Double]], y: Array[Int]): (Int, ClassificationValidations[KNN[Array[Double]]]) =
    (1 until 30).map(k => (k, my_knn(x, y, k))).maxBy(_._2.avg.accuracy)

  def my_svm(x_train: Array[Array[Double]], x_test: Array[Array[Double]],
             y_train: Array[Int], y_test: Array[Int], params: Properties): Double = {
    val trainer: BiFunction[Array[Array[Double]], Array[Int], Classifier[Array[Double]]] = { case (x, y) =>
      SVM.fit(x, y, params)
    }
    val model = OneVersusOne.fit(x_train, y_train, trainer)
    val prediction = model.predict(x_test)
    val err = Error.of(y_test, prediction)
    1 - err.toDouble / y_test.length
  }

  def my_best_svm(x: Array[Array[Double]], y: Array[Int]): List[(Properties, Double)] = {
    val (x_train, x_test) = split_train_test(x)
    val (y_train, y_test) = split_train_test(y)
    val hp = new Hyperparameters()
      .add("smile.svm.kernel", "linear" +: (1 until 5).toArray.map(x => s"Gaussian(${x.toDouble / 4})"))
      .add("smile.svm.C", (1 until 20).toArray.map(x => x.toDouble / 2))
      .add("smile.svm.epochs", (1 to 3).toArray)
    val bestModels = hp.grid().toArray(k => new Array[Properties](k)).map { params =>
      (params, Try(my_svm(x_train, x_test, y_train, y_test, params)))
    }.filter(_._2.isSuccess).map(e => (e._1, e._2.get)).sortBy(_._2)(Ordering.Double.TotalOrdering.reverse)
    val bestAccuracy: Double = bestModels(0)._2
    bestModels.takeWhile(e => e._2 == bestAccuracy).toList.map(e => (e._1, e._2))
  }

  def parzen_window(x: Array[Array[Double]], y: Array[Int],
                    k: Int, step: Double): ClassificationValidations[KNN[Array[Double]]] = {
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

  def my_best_parzen_window(x: Array[Array[Double]], y: Array[Int]): (Int, Double, ClassificationValidations[KNN[Array[Double]]]) = {
    val ks = (1 until 30).toArray
    val steps = (1 until 40).toArray.map(_ * 0.1)
    (for (k <- ks; s <- steps; m <- Try(parzen_window(x, y, k, s)).toOption.toList)
      yield (k, s, m)
      ).maxBy(_._3.avg.accuracy)
  }

  def main(args: Array[String]): Unit = {
    val wineCsv = Paths.get(getClass.getClassLoader.getResource("wine.data").toURI).toFile
    val wine: DataFrame = read.csv(wineCsv.getAbsolutePath, header = false)
    val (x, y): (Array[Array[Double]], Array[Int]) = extract_x_y(wine)
    MathEx.normalize(x)
    println("best k-NN: " + my_best_knn(x, y)) // k=3, accuracy=96.81% ± 3.44
    println("best SVM: " + my_best_svm(x, y)) // sigma=1.5, regulation=6.0, accuracy=1.0
    println("best Parzen window: " + my_best_parzen_window(x, y)) // k=3, step=0.8, accuracy=97.65% ± 4.11
  }
}