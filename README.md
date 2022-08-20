# ML Test
Test repository for machine learning with [Smile](https://github.com/haifengl/smile).
_ML Test_ currently contains few classification and 1 regression method on few datasets.
All datasets loaded from [ML archive](https://archive.ics.uci.edu/ml/index.php).

## About Smile
Why Smile?
Smile is well-documented and relatively fast JVM library for Data Science.

There are few alternatives:
- Python
- Scala Spark
- R

Python is de-facto standard in ML because of high speed (C/C++ under the hood) and big community, but
despite that:
- it is not a type-safe language,
- some functions in libs are poorly documented.

Spark in also popular and with good community, but Spark ML misses most of ML algorithms and also Spark is not suitable
for small datasets.

R is not so popular and I don't have big background in it, so I just skip it.

So for me Smile was much easier to learn and to use.