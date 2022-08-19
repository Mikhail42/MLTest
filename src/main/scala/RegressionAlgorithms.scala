package org.ionkin.ml.test

import smile.hpo.Hyperparameters
import smile.regression.SVM
import smile.validation.{LOOCV, RegressionMetrics}

import java.util.Properties
import scala.util.Try

object RegressionAlgorithms {

  def optimize[X, Y](x: X, y: Y, hp: Hyperparameters,
                     opt: (X, Y, Properties) => RegressionMetrics): List[(Properties, Double)] = {
    val bestModels = hp.grid().toArray(k => new Array[Properties](k)).map { params =>
      (params, Try(opt(x, y, params)))
    }.filter(_._2.isSuccess).map(e => (e._1, e._2.get.mse)).sortBy(_._2)
    val bestAccuracy: Double = bestModels(0)._2
    bestModels.takeWhile(e => e._2 == bestAccuracy).toList.map(e => (e._1, e._2))
  }

  def my_svm(x: Array[Array[Double]], y: Array[Double], params: Properties): RegressionMetrics = {
    LOOCV.regression(x, y, (x: Array[Array[Double]], y: Array[Double]) =>
      SVM.fit(x, y, params))
  }

  def my_best_svm(x: Array[Array[Double]], y: Array[Double]): List[(Properties, Double)] = {
    val hp = new Hyperparameters()
      .add("smile.svm.kernel", "linear" +: (1 until 10).toArray.map(x => s"Gaussian(${x * 0.3})"))
      .add("smile.svm.epsilon", Array(1, 3, 10, 30, 100))
      .add("smile.svm.C", Array(1, 3, 10, 30, 100))
    optimize(x, y, hp, my_svm)
  }
}
