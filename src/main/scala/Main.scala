package org.ionkin.ml.test

import smile.classification.*
import smile.data.DataFrame
import smile.math.MathEx
import smile.validation.ClassificationValidations
import smile.{data, read}

import java.nio.file.Paths

def my_knn(data: DataFrame): Unit =
  val y: Array[Int] = data.intVector(0).toIntArray.map(x => x-1)
  val x: Array[Array[Double]] = data.drop(0).toArray
  MathEx.normalize(x)
  val startTime = System.currentTimeMillis()
  val kFold = 10
  val metrics: ClassificationValidations[KNN[Array[Double]]] =
    smile.validation.cv.classification(kFold, x, y) { case (x, y) => knn(x, y, 5) }
  println(System.currentTimeMillis() - startTime + " ms")
  println(metrics)

@main def main(): Unit =
  val wineCsv = Paths.get(getClass.getClassLoader.getResource("wine.data").toURI).toFile
  val wine: DataFrame = read.csv(wineCsv.getAbsolutePath)
  my_knn(wine)
