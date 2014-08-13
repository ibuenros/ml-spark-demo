package ibuenros.mllibdemos

import org.apache.spark.rdd.RDD
import org.apache.spark.SparkContext._
import org.apache.spark.mllib.linalg.{Vectors, Vector}
import scala.util.Random
import org.apache.spark.Logging

object KMeans extends Logging {

  def euclideanDistSquared(a: Vector, b: Vector): Double = {
    val pairs = a.toArray.zip(b.toArray)
    val squaredSum = pairs.map({ case(a,b) => (a-b)*(a-b) }).sum
    squaredSum
  }

  def train(points: RDD[Vector], k: Int, iterations: Int): Array[Vector] = {

    points.persist()

    val maxSize = points.map(_.size).max
    val minSize = points.map(_.size).min
    assert(maxSize == minSize)

    val maxBounds = points.map(_.toArray).reduce { (a, b) =>
      a.zip(b).map(entry => math.max(entry._1, entry._2))
    }
    val minBounds = points.map(_.toArray).reduce { (a, b) =>
      a.zip(b).map(entry => math.min(entry._1, entry._2))
    }
    val bounds = maxBounds.zip(minBounds)

    val rand = new Random()
    val seeds = (0 until k).map { i =>
      Vectors.dense(bounds.map { case(max, min) =>
        rand.nextDouble()*(max-min) + min
      })
    }

    logInfo("-------SEEDS---------")
    seeds.foreach(c => logInfo(c.toArray.mkString(",")))

    var centers = seeds.toArray

    (0 until iterations).foreach { i =>
      val clusteredPoints = points.map({ p =>
        val distances = centers.map(KMeans.euclideanDistSquared(_,p))
        val closest = distances.zipWithIndex.minBy(_._1)._2
        (closest, p)
      }).persist()

      val sumsByKey = clusteredPoints.reduceByKey({ (vecA, vecB) =>
        Vectors.dense(vecA.toArray.zip(vecB.toArray).map({case (a,b) => a + b}))
      }).collect().toMap
      val pointsByKey = clusteredPoints.countByKey()

      centers = sumsByKey.keys.map({ k =>
        Vectors.dense(sumsByKey(k).toArray.map(  _ / pointsByKey(k)))
      }).toArray

      clusteredPoints.unpersist(false)
    }

    centers
  }
}
