package org.ionkin.ml.test

import breeze.linalg.*
import breeze.numerics.*
import breeze.stats.mean
import org.apache.commons.math3.ml.clustering.*

import java.nio.file.Paths
import java.util.Random

type Matrix = DenseMatrix[Double]
type Vector = DenseVector[Double]

def test_train_split(X: Matrix, Y: Vector, testSize: Int): (Matrix, Matrix, Vector, Vector) =
  val testIndexes: Array[Int] = new Random().ints(testSize, 0, testSize).toArray
  val X_train = X.delete(testIndexes, Axis._0)
  //val Y_train =
  ???


def kMean(X_train: Matrix, Y_train: Vector, k: Int = 5) =
  val clusterer = new FuzzyKMeansClusterer[Vector](k)
  //clusterer.cluster(X_train)
  ???


@main def main(): Unit =
  val wineCsv = Paths.get(getClass.getClassLoader.getResource("wine.data").toURI).toFile
  val wine: DenseMatrix[Double] = breeze.linalg.csvread(wineCsv)
  val X = wine.delete(0, Axis._1)
  val Y = wine(::, 0)
  val trainSize = (X.size * 0.8).toInt
  val testSize = X.size - trainSize

  println(mean(X))
  println(sin(Y))
  println("Hello, world!")

