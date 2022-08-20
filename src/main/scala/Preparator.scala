package org.ionkin.ml.test

import com.typesafe.scalalogging.StrictLogging
import smile.data.DataFrame
import smile.data.`type`.StructField
import smile.feature.imputation.SVDImputer

import java.util.Random
import scala.reflect.ClassTag
import scala.util.Try

object Preparator extends StrictLogging {
  def classification_extract_y(data: DataFrame, y_column_id: Int): Array[Int] = {
    val y_schema: StructField = data.schema().field(y_column_id)
    val y0: Array[Int] = y_schema.`type` match {
      case t if t.isDouble => data.doubleVector(y_column_id).toIntArray
      case t if t.isInt => data.intVector(y_column_id).array()
    }
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
    var x = extract_x(x_data)
    if (has_nan(x)) {
      val k = Math.min(x.length, x(0).length) - 1 // TODO: the number of eigenvectors used for imputation
      x = SVDImputer.impute(x, k, 5)
    }
    (x, y)
  }

  private def has_nan(ar: Array[Double]): Boolean = ar.exists(_.isNaN)

  private def has_nan(mat: Array[Array[Double]]): Boolean = mat.exists(has_nan)

  private def convert_x_value(d: Object): Double = d match {
    case d: java.lang.Double => d
    case i: java.lang.Integer => i.toDouble
    case l: java.lang.Long => l.toDouble
    case f: java.lang.Float => f.toDouble
    case c: java.lang.Character if c == '?' => Double.NaN
    case c: java.lang.Character =>
      val lower = Character.toLowerCase(c)
      if (lower.isDigit) lower.getNumericValue
      else if (lower - 'a' >= 0 && 'z' - lower >= 0) (lower - 'a').toDouble
      else throw new RuntimeException(s"Can't convert $c to double")
    case s: java.lang.String if s.length == 1 => convert_x_value(Character.valueOf(s.charAt(0)))
    case s: java.lang.String if Try(s.toDouble).isSuccess => s.toDouble
    case x => throw new RuntimeException(s"Can't convert $x to double")
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
        x_double_data(i)(k) =
          try convert_x_value(x_data.get(i, k))
          catch {
            case exc: Exception =>
              logger.error(s"Can't convert X[$i][$k] to double", exc)
              throw exc
          }
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
