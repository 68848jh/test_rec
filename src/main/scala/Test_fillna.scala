
import org.apache.spark.ml.feature.Imputer
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.types.StructType


object Test_fillna {
  case class Rating(id: Int,
                    userId: String,
                    itemId: String,
                    rating: Float)

  def main(args: Array[String]): Unit = {

    val spark = SparkSession.builder()
      .master("local")
      .getOrCreate()

    val data = spark.read.format("csv")
      .option("header", true)
      .option("delimiter", ",")
      .option("inferSchema", true)
      .schema(ScalaReflection.schemaFor[Rating]
        .dataType.asInstanceOf[StructType])
      .load("E:\\fillna.csv")

    //      data.show(false)

    val imputer = new Imputer()
      .setInputCols(Array("rating"))//输入的列名
      .setOutputCols(Array("out_rating")) //输出的列名
      .setMissingValue(Float.NaN) //要取代的缺失值
      .setStrategy("mean") //填充缺失值的办法


    val model = imputer.fit(data)
    val data2 = model.transform(data)
    data2.show(false)
    spark.stop()


  }

}
