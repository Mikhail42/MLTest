package org.ionkin.ml.test

import java.nio.file.Paths
import java.util.Random

@main def main(): Unit =
  val wineCsv = Paths.get(getClass.getClassLoader.getResource("wine.data").toURI).toFile


