package org.ionkin.ml.test

import java.nio.file.Paths
import smile.classification.*
import smile.data.formula.FormulaBuilder
import smile.read
import smile.data.formula.FormulaBuilder.*
import smile.data.formula.Terms

import scala.collection.mutable.ListBuffer

@main def main(): Unit =
  val wineCsv = Paths.get(getClass.getClassLoader.getResource("wine.data").toURI).toFile
  val wine = read.csv(wineCsv.getAbsolutePath)
  val formula = FormulaBuilder(Option(Terms.$("1")), ListBuffer()).toFormula
  val model = randomForest(formula, wine)
  println(model.metrics)
