package org.ionkin.ml.test

import smile.hpo.Hyperparameters
import smile.validation.ClassificationValidations

import java.util.Properties
import scala.util.Try


object Optimizer {

  def optimize[X, Y, M](x: X, y: Y, hp: Hyperparameters,
                        opt: (X, Y, Properties) => ClassificationValidations[M]): List[(Properties, Double)] = {
    val bestModels = hp.grid().toArray(k => new Array[Properties](k)).map { params =>
      (params, Try(opt(x, y, params)))
    }.filter(_._2.isSuccess).map(e => (e._1, e._2.get.avg.accuracy)).sortBy(_._2)(Ordering.Double.TotalOrdering.reverse)
    val bestAccuracy: Double = bestModels(0)._2
    bestModels.takeWhile(e => e._2 == bestAccuracy).toList.map(e => (e._1, e._2))
  }
}
