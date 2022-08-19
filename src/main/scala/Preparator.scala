package org.ionkin.ml.test

import smile.data.DataFrame

import java.util.Random
import scala.reflect.ClassTag
import scala.util.Try

object Preparator {
  def classification_extract_y(data: DataFrame, y_column_id: Int): Array[Int] = {
    val y0: Array[Int] = data.doubleVector(y_column_id).toIntArray
    // Assume that classes are 2 and 4. Then it will be map (2->0, 4->1)
    val y_to_y_id: Map[Int, Int] = y0.toSet.toSeq.sorted.zipWithIndex.toMap
    val y = y0.map(y_to_y_id)
    y
  }

  def regression_extract_y(data: DataFrame, y_column_id: Int): Array[Double] =
    data.doubleVector(y_column_id).array()

  def regression_extract_x_y(data: DataFrame, y_column_id: Int): (Array[Array[Double]], Array[Double]) = {
    val y = regression_extract_y(data, y_column_id)
    val x_data = data.drop(y_column_id)
    val x = extract_x(x_data)
    (x, y)
  }

  def classification_extract_x_y(data: DataFrame, y_column_id: Int): (Array[Array[Double]], Array[Int]) = {
    val y = classification_extract_y(data, y_column_id)
    val x_data = data.drop(y_column_id)
    val x = extract_x(x_data)
    (x, y)
  }

  def extract_x(x_data: DataFrame): Array[Array[Double]] = {
    // I need to handle 2 cases:
    // 1. when there is no column scheme with double. In this case I can just fix csv and convert 1st row to doubles
    // 2. when there is missing cells in dataset. In this case I can:
    // -- drop row, or
    // -- use predicted value, or
    // -- use algorithms that tolerant to missing values, like k-NN
    val x_double_data: Array[Array[Double]] = Array.ofDim[Double](x_data.size(), x_data.ncol())
    for (k <- 0 until x_data.ncol()) {
      for (i <- 0 until x_data.size()) {
        val d: Double = Try {
          val converted: Double = x_data.get(i, k) match {
            case i: java.lang.Integer => i.toDouble
            case l: java.lang.Long => l.toDouble
            case d: java.lang.Double => d
            case c: java.lang.Character => (Character.toLowerCase(c) - 'a').toInt.toDouble
            case s: java.lang.String if s.length == 1 => (Character.toLowerCase(s.charAt(0)) - 'a').toInt.toDouble
            case _ => Double.NaN
          }
          converted
        }.getOrElse(Double.NaN)
        x_double_data(i)(k) = d
      }
    }
    x_double_data
  }

  def get_test_indexes(testSize: Int): Set[Int] =
    new Random((2L >> 31) / 42).ints(testSize, 0, testSize).toArray.toSet

  def split_train_test[T: ClassTag](ar: Array[T]): (Array[T], Array[T]) = {
    val testSize = (ar.length * 0.3).toInt
    val testIndexes: Set[Int] = get_test_indexes(testSize)
    ar.zipWithIndex.partition { case (el, i) => !testIndexes.contains(i) } match {
      case (tr, ts) => (tr.map(_._1), ts.map(_._1))
    }
  }
}
