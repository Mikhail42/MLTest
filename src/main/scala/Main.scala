package org.ionkin.ml.test

import com.typesafe.scalalogging.StrictLogging
import smile.classification.{Classifier, KNN, MLP, OneVersusOne, RDA, SVM}
import smile.data.DataFrame
import smile.hpo.Hyperparameters
import smile.math.MathEx
import smile.math.distance.{EuclideanDistance, Metric}
import smile.{classification, read}
import smile.validation.ClassificationValidations
import smile.validation.metric.Error

import java.io.File
import java.nio.file.Paths
import java.util.function.BiFunction
import java.util.{Properties, Random}
import scala.reflect.ClassTag
import scala.util.Try

object Main extends StrictLogging {

  def extract_x_y(data: DataFrame, y_column_id: Int): (Array[Array[Double]], Array[Int]) = {
    val y0: Array[Int] = data.intVector(y_column_id).toIntArray
    // Assume that classes are 2 and 4. Then it will be map (2->0, 4->1)
    val y_to_y_id: Map[Int, Int] = y0.toSet.toSeq.sorted.zipWithIndex.toMap
    val y = y0.map(y_to_y_id)
    val x: Array[Array[Double]] = data.drop(y_column_id).toArray()
    (x, y)
  }

  def get_test_indexes(testSize: Int): Set[Int] =
    new Random(42).ints(testSize, 0, testSize).toArray.toSet

  def split_train_test[T : ClassTag](ar: Array[T]): (Array[T], Array[T]) = {
    val testSize = (ar.length * 0.3).toInt
    val testIndexes: Set[Int] = get_test_indexes(testSize)
    ar.zipWithIndex.partition { case (el, i) => !testIndexes.contains(i) } match {
      case (tr, ts) => (tr.map(_._1), ts.map(_._1))
    }
  }

  def my_knn(x: Array[Array[Double]], y: Array[Int], k: Int): ClassificationValidations[KNN[Array[Double]]] = {
    smile.validation.cv.classification(k = 10, x, y) { case (x, y) => classification.knn(x, y, k = k) }
  }

  def my_best_knn(x: Array[Array[Double]], y: Array[Int]): (Int, ClassificationValidations[KNN[Array[Double]]]) =
    (1 until 30).map(k => (k, my_knn(x, y, k))).maxBy(_._2.avg.accuracy)

  def my_svm(x_train: Array[Array[Double]], x_test: Array[Array[Double]],
             y_train: Array[Int], y_test: Array[Int], params: Properties): Double = {
    val classes = y_train.toSet.size
    val model: Classifier[Array[Double]] = if (classes > 2) {
      val trainer: BiFunction[Array[Array[Double]], Array[Int], Classifier[Array[Double]]] = {
        case (x, y) =>
          SVM.fit(x, y, params)
      }
      OneVersusOne.fit(x_train, y_train, trainer)
    } else {
      SVM.fit(x_train, y_train, params)
    }
    val prediction = model.predict(x_test)
    val err = Error.of(y_test, prediction)
    val accuracy = 1 - err.toDouble / y_test.length
    //logger.debug(s"accuracy=$accuracy for SVM with params=$params")
    accuracy
  }

  def my_best_svm(x: Array[Array[Double]], y: Array[Int]): List[(Properties, Double)] = {
    val (x_train, x_test) = split_train_test(x)
    val (y_train, y_test) = split_train_test(y)
    val hp = new Hyperparameters()
      .add("smile.svm.kernel", "linear" +: (1 until 5).toArray.map(x => s"Gaussian(${x.toDouble / 4})"))
      .add("smile.svm.C", (1 until 20).toArray.map(x => x.toDouble / 2))
      .add("smile.svm.epochs", (1 to 3).toArray)
    // (5+1)*20*3=120*3=360 iterations, ~0.75 seconds per iter for spamdata, total ~4.5 minutes for spam
    val bestModels = hp.grid().toArray(k => new Array[Properties](k)).map { params =>
      (params, Try(my_svm(x_train, x_test, y_train, y_test, params)))
    }.filter(_._2.isSuccess).map(e => (e._1, e._2.get)).sortBy(_._2)(Ordering.Double.TotalOrdering.reverse)
    val bestAccuracy: Double = bestModels(0)._2
    bestModels.takeWhile(e => e._2 == bestAccuracy).toList.map(e => (e._1, e._2))
  }

  def parzen_window(x: Array[Array[Double]], y: Array[Int],
                    k: Int, step: Double): ClassificationValidations[KNN[Array[Double]]] = {
    val weightedDistance = new Metric[Array[Double]] {
      def core(el: Double): Double = if (math.abs(el) < 1) 1 - el * el else 0

      def d(x1: Array[Double], x2: Array[Double]): Double = {
        val weight = new Array[Double](x1.length)
        for (i <- weight.indices) {
          weight(i) = core((x1(i) - x2(i)) / step)
        }
        val distance = new EuclideanDistance(weight)
        distance.d(x1, x2)
      }
    }
    val res = smile.validation.cv.classification(k = 10, x, y) { case (x, y) => classification.knn(x, y, k = k, weightedDistance) }
    //logger.debug(s"parzen_window with k=$k, step=$step, accuracy=${res.avg.accuracy}")
    res
  }

  def my_best_parzen_window(x: Array[Array[Double]], y: Array[Int]): (Int, Double, ClassificationValidations[KNN[Array[Double]]]) = {
    val ks = (1 until 10).map(_ * 2 - 1).toArray
    val steps = (1 until 10).toArray.map(_ * 0.4)
    (for (k <- ks; s <- steps; m <- Try(parzen_window(x, y, k, s)).toOption.toList)
      yield (k, s, m)
      ).maxBy(_._3.avg.accuracy)
  }

  def my_mlp(x: Array[Array[Double]], y: Array[Int], props: Properties): ClassificationValidations[MLP] = {
    val res = smile.validation.cv.classification(math.min(y.length / 10, 20), x, y) { case (x, y) =>
      val mlp = MLP.fit(x, y, props)
      mlp
    }
    //logger.debug(s"MLP accuracy=${res.avg.accuracy} for params=$props")
    res
  }

  def my_best_mlp(x: Array[Array[Double]], y: Array[Int]): List[(Properties, Double)] = {
    val hp = new Hyperparameters()
      .add("smile.mlp.layers", (1 until 5).toArray.map(x => s"ReLU(${(x-1)*20+1})") ++
        (1 until 5).toArray.map(x => s"Sigmoid(${(x-1)*20+1})"))
      .add("smile.mlp.mini_batch", Array(1, 3, 10, 30, 100))
      .add("smile.mlp.epochs", Array(1, 3, 10, 30, 100))
    optimize(x, y, hp, my_mlp)
  }

  def my_rda(x: Array[Array[Double]], y: Array[Int], props: Properties): ClassificationValidations[RDA] =
    smile.validation.cv.classification(math.min(y.length / 10, 20), x, y) { case (x, y) =>
      RDA.fit(x, y, props)
    }

  def my_best_rda(x: Array[Array[Double]], y: Array[Int]): List[(Properties, Double)] = {
    val hp = new Hyperparameters()
      .add("smile.rda.alpha", (0 to 10).map(_ * 0.1).toArray)
    optimize(x, y, hp, my_rda)
  }

  def optimize[X, Y, M](x: X, y: Y, hp: Hyperparameters,
               opt: (X, Y, Properties) => ClassificationValidations[M]): List[(Properties, Double)] = {
    val bestModels = hp.grid().toArray(k => new Array[Properties](k)).map { params =>
      (params, Try(opt(x, y, params)))
    }.filter(_._2.isSuccess).map(e => (e._1, e._2.get.avg.accuracy)).sortBy(_._2)(Ordering.Double.TotalOrdering.reverse)
    val bestAccuracy: Double = bestModels(0)._2
    bestModels.takeWhile(e => e._2 == bestAccuracy).toList.map(e => (e._1, e._2))
  }

  def show_best(path: File, y_column_id: Int): Unit = {
    val data: DataFrame = read.csv(path.getAbsolutePath, header = false)
    val y_col_id = if (y_column_id >= 0) y_column_id else data.ncol() + y_column_id
    val (x, y): (Array[Array[Double]], Array[Int]) = extract_x_y(data, y_col_id)
    MathEx.normalize(x)
    logger.info("best RDA: " + my_best_rda(x, y))
    logger.info("best MLP: " + my_best_mlp(x, y))
    logger.info("best k-NN: " + my_best_knn(x, y))
    logger.info("best SVM: " + my_best_svm(x, y))
    logger.info("best Parzen window: " + my_best_parzen_window(x, y))
  }

  def wine(): Unit = {
    val winePath = Paths.get(getClass.getClassLoader.getResource("wine.csv").toURI).toFile
    show_best(winePath, y_column_id = 0)
    // accuracy=99.4% for RDA for alpha=0.1:0.1:1 (maybe with few exceptions)
    // MLP accuracy=0.9339869281045754 for epochs=100, layers=ReLU(41), mini_batch=1
    // k=3, accuracy=96.81% ± 3.44 for k-NN
    // sigma=1.5, regulation=6.0, accuracy=1.0 (100%)  for SVM
    // k=3, step=0.8, accuracy=97.65% ± 4.11 for Parzen window
  }

  def spam(): Unit = {
    val spamPath = Paths.get(getClass.getClassLoader.getResource("spambase.csv").toURI).toFile
    show_best(spamPath, y_column_id = -1)
    // RDA: accuracy=89.65% for alpha=0.1. very fast optimization
    // MLP accuracy=0.9200178806700545 for epochs=100, layers=ReLU(21), mini_batch=1,
    // optimization for MLP can be really slow (1-30 seconds per iteration)
    // k=1, accuracy=91.33% ± 1.29 for k-NN
    // sigma=0.25, regulation=9.5, epochs=2, accuracy=0.838 (83.8%) for SVM
    // k=1, step=1.2, accuracy=91.68% ± 1.05 for Parzen window
  }

  def main(args: Array[String]): Unit = {
    logger.info("start")
    //wine()
    spam()
  }
}