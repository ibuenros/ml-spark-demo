package ibuenros.mllibdemos

import org.apache.spark.mllib.linalg.{Vector, Vectors}
import scala.util.Random
import org.apache.spark.{SparkConf, SparkContext}

object KMeansMain {
  def main(args: Array[String]) {

    val rand = new Random()
    val sigma = 0.5
    val pointsPerCluster = 100

    val centers = Array(Vectors.dense(1,1), Vectors.dense(-1,1), Vectors.dense(-1,-1), Vectors.dense(1,-1))

    val points = centers.flatMap{ center =>
      (0 until (rand.nextInt(pointsPerCluster) + pointsPerCluster)).map { i =>
        Vectors.dense(center.toArray.map(i => i + rand.nextGaussian()*sigma))
      }
    }

    //println("-------ORIGINAL POINTS---------")
    //points.foreach(c => println(c.toArray.mkString(",")))

    val conf = new SparkConf().
      setAppName("KMeans")
    val sc = new SparkContext(conf)

    val pointsRDD = sc.parallelize(points)

    val centersPredicted = KMeans.train(pointsRDD, 4, 100)

    centersPredicted.foreach(c => println(c.toArray.mkString(",")))

  }
}
