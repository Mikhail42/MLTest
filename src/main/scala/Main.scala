package org.ionkin.ml.test

import smile.classification.*
import smile.data.DataFrame
import smile.math.MathEx
import smile.math.kernel.GaussianKernel
import smile.validation.ClassificationValidations
import smile.validation.metric.Error
import smile.{data, read}

import java.nio.file.Paths
import java.util.Random
import scala.reflect.ClassTag


def extract_x_normalized_y(data: DataFrame) =
  val y: Array[Int] = data.intVector(0).toIntArray.map(x => x - 1)
  val x: Array[Array[Double]] = data.drop(0).toArray
  MathEx.normalize(x)
  (x, y)

def get_test_indexes(testSize: Int): Set[Int] =
  new Random(42).ints(testSize, 0, testSize).toArray.toSet

def split_train_test[T : ClassTag](ar: Array[T], testSize: Int): (Array[T], Array[T]) =
  val testIndexes: Set[Int] = get_test_indexes(testSize)
  ar.zipWithIndex.partition { case (el, i) => !testIndexes.contains(i) } match
    case (tr, ts) => (tr.map(_._1), ts.map(_._1))

def my_knn(data: DataFrame, k: Int) =
  val (x, y) = extract_x_normalized_y(data)
  smile.validation.cv.classification(k=10, x, y) { case (x, y) => knn(x, y, k=k) }

def my_best_knn(data: DataFrame) =
  (1 until 30).map(k => (k, my_knn(data, k))).maxBy(_._2.avg.accuracy)

def my_svm(data: DataFrame, sigma: Double, regulation: Double) =
  val (x, y) = extract_x_normalized_y(data)
  val testSize = (x.length * 0.3).toInt
  val (x_train, x_test) = split_train_test(x, testSize)
  val (y_train, y_test) = split_train_test(y, testSize)
  val kernel = new GaussianKernel(sigma)
  val model = OneVersusOne.fit(x_train, y_train, (x_i, y) => svm(x_i, y, kernel, regulation))
  val prediction = model.predict(x_test)
  val err = Error.of(y_test, prediction)
  1 - err.toDouble / y_test.length

def my_best_svm(data: DataFrame) =
  val svm_accuracies = for {
    sigma <- (1 until 20).map(x => x.toDouble / 2)
    regulation <- (1 until 20).map(x => x.toDouble / 2)
  } yield (sigma, regulation, my_svm(data, sigma, regulation))
  svm_accuracies.maxBy(_._3)

@main def main(): Unit =
  val wineCsv = Paths.get(getClass.getClassLoader.getResource("wine.data").toURI).toFile
  val wine: DataFrame = read.csv(wineCsv.getAbsolutePath)
  println("best k-NN: " + my_best_knn(wine))
  println("best SVM: " + my_best_svm(wine))