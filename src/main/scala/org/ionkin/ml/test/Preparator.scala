package org.ionkin.ml.test

import smile.data.DataFrame

object Preparator {
  def extract_x_y(data: DataFrame, y_column_id: Int): (Array[Array[Double]], Array[Int]) = {
    val y0: Array[Int] = data.intVector(y_column_id).toIntArray
    // Assume that classes are 2 and 4. Then it will be map (2->0, 4->1)
    val y_to_y_id: Map[Int, Int] = y0.toSet.toSeq.sorted.zipWithIndex.toMap
    val y = y0.map(y_to_y_id)
    val x_data = data.drop(y_column_id)
    // I need to handle 2 cases:
    // 1. when there is no column scheme with double. In this case I can just fix csv and convert 1st row to doubles
    // 2. when there is missing cells in dataset. In this case I can:
    // -- drop row, or
    // -- use predicted value, or
    // -- use algorithms that tolerant to missing values, like k-NN
    val x_double_data: Array[Array[Double]] = Array.ofDim[Double](x_data.size(), x_data.ncol())
    for (k <- 0 until x_data.ncol()) {
      for (i <- 0 until x_data.size()) {
        x_double_data(i)(k) = Try(x_data.get(i, k).toString.toDouble).getOrElse(Double.NaN)
      }
    }
    (x_double_data, y)
  }
}
