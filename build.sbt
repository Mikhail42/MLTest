ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.8"

lazy val breezeVersion = "2.0"
libraryDependencies += "org.scalanlp" %% "breeze" % breezeVersion
libraryDependencies += "org.scalanlp" %% "breeze-viz" % breezeVersion
resolvers += "Sonatype Releases" at "https://oss.sonatype.org/content/repositories/releases/"

lazy val root = (project in file("."))
  .settings(
    name := "MLTest",
    idePackagePrefix := Some("org.ionkin.ml.test")
  )
