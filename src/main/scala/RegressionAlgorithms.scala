package org.ionkin.ml.test

import com.typesafe.scalalogging.StrictLogging
import org.ionkin.ml.test.Preparator.regression_extract_x_y
import smile.data.DataFrame
import smile.hpo.Hyperparameters
import smile.math.MathEx
import smile.read
import smile.regression.SVM
import smile.validation.{LOOCV, RegressionMetrics}

import java.io.File
import java.util.Properties
import scala.util.Try

object RegressionAlgorithms extends StrictLogging {

  def optimize[X, Y](x: X, y: Y, hp: Hyperparameters,
                     opt: (X, Y, Properties) => RegressionMetrics): List[(Properties, Double)] = {
    val bestModels = hp.grid().toArray(k => new Array[Properties](k)).map { params =>
      (params, Try(opt(x, y, params)))
    }.filter(_._2.isSuccess).map(e => (e._1, e._2.get.mse)).sortBy(_._2)
    val bestAccuracy: Double = bestModels(0)._2
    bestModels.takeWhile(e => e._2 == bestAccuracy).toList.map(e => (e._1, e._2))
  }

  def my_svm(x: Array[Array[Double]], y: Array[Double], params: Properties): RegressionMetrics = {
    val err = LOOCV.regression(x, y, (x: Array[Array[Double]], y: Array[Double]) => SVM.fit(x, y, params))
    logger.debug(s"SVM mse=${err.mse} for $params")
    err
  }

  def my_best_svm(x: Array[Array[Double]], y: Array[Double]): List[(Properties, Double)] = {
    val hp = new Hyperparameters()
      .add("smile.svm.kernel", "linear" +: (1 until 10).toArray.map(x => s"Gaussian(${x * 0.1})"))
      .add("smile.svm.epsilon", Array(0.1, 1, 3, 10, 30, 100).map(_ * 0.1))
      .add("smile.svm.C", Array(1, 3, 10, 30, 100, 300).map(_ * 0.1))
    optimize(x, y, hp, my_svm)
  }

  def show_best(path: File, y_column_id: Int): Unit = {
    val data: DataFrame = read.csv(path.getAbsolutePath, header = false)
    val y_col_id = if (y_column_id >= 0) y_column_id else data.ncol() + y_column_id
    val (x, y): (Array[Array[Double]], Array[Double]) = regression_extract_x_y(data, y_col_id)
    MathEx.normalize(x)
    logger.info("best SVM: " + my_best_svm(x, y))
  }
}
