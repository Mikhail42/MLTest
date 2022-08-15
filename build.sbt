ThisBuild / version := "0.1.0-SNAPSHOT"

ThisBuild / scalaVersion := "2.13.8"

// close from github, change version to snapshot and publish local
ThisBuild / resolvers += Resolver.mavenLocal
lazy val smile = "com.github.haifengl" %% "smile-scala" % "3.0.0-SNAPSHOT"

// https://github.com/haifengl/smile
def allOsClassifiers(moduleId: ModuleID): ModuleID =
  moduleId classifier "macosx-x86_64" classifier "windows-x86_64" classifier "linux-x86_64" classifier "linux-arm64" classifier "linux-ppc64le" classifier "android-arm64" classifier "ios-arm64"

def linuxX86_64Classifier(moduleId: ModuleID): ModuleID =
  moduleId classifier "linux-x86_64"

def withOSClassifier(moduleId: ModuleID): ModuleID = linuxX86_64Classifier(moduleId)

lazy val byteDecoLibs = Seq(
  withOSClassifier("org.bytedeco" % "javacpp"   % "1.5.7"),
  withOSClassifier("org.bytedeco" % "openblas"  % "0.3.19-1.5.7"),
  withOSClassifier("org.bytedeco" % "arpack-ng" % "3.8.0-1.5.7")
)

libraryDependencies ++= Seq(smile) ++ byteDecoLibs

lazy val root = (project in file("."))
  .settings(
    name := "MLTest",
    idePackagePrefix := Some("org.ionkin.ml.test")
  )
