package org.ionkin.ml.test

object Util {

  def has_nan(ar: Array[Double]): Boolean = ar.exists(_.isNaN)

  def has_nan(mat: Array[Array[Double]]): Boolean = mat.exists(has_nan)

}
