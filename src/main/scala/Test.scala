import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.catalyst.ScalaReflection
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types.StructType
object Test {

  case class UserItem(userId:String,itemId:String,score:Float)  //指定数据类型
  def  main(array: Array[String]): Unit = {

    /**
     *  基于用户的协同过滤有以下几个缺点：
     *  1. 如果用户量很大，计算量就很大
     *  2. 用户-物品打分矩阵是一个非常非常非常稀疏的矩阵，会面临大量的null值
     *  很难得到两个用户的相似性
     *  3. 会将整个用户-物品打分矩阵都载入到内存，而往往这个用户-物品打分矩阵是一个
     *  非常大的矩阵
     *
     *  所以通常不太建议使用基于用户的协同过滤
     *
     * */
    //导入数据
    val spark = SparkSession.builder()
      .master(master = "local") //本地运行
      .getOrCreate()

    val df = spark.read.format(source = "csv") //导入csv文件将它生成dataframe列表
      .option("header", true)
      .option("delimiter", ",") //以，作为分割
      .option("inferschema", true) //数据类型是否要推断
      .schema(ScalaReflection.schemaFor[UserItem] //UserItem类型指定
        .dataType.asInstanceOf[StructType])
      .load(path = "E:\\cf_user_based.csv") //文件路径

    df.show(false)

    //通过余弦相似度计算用户的相似度 余弦相似度的公式 ： (A * B) / (|A| * |B|)
    //分母 每个向量的模的乘积
    import spark.implicits._
    //将dataframe表格的数据转化成rdd，然后通过map转化成键值对的rdd
    val df_score_mod = df.rdd.map(x => (x(0).toString,x(2).toString))  // key:x(0)  value:x(2)
      .groupByKey()             //按照用户id进行分组
      .mapValues(score => math.sqrt(      //对每个商品的评分进行平方在+和最后开方
        score.toArray.map(itemScore => math.pow(itemScore.toDouble,2)
        ).reduce(_+_)
        // ((物品a的打分)**2 + (物品b的打分)**2 .. (物品n的打分)**2))** 1/2
      ))
      .toDF("userId","mod")  //新的dataframe表格 第一列是userId，第二列是模

    df_score_mod.show(false)
    df.printSchema() //打印表结构

    //分子
    val _dfTmp = df.select(
      col("userId").as("_userId"), //改名
      col("itemId"),
      col("score").as("_score")  //改名
    )
    //这里目的是把两两用户都放到了同一张表里
    val _df = df.join(_dfTmp,df("itemId") === _dfTmp("itemId"))
      .where(
        df("userId") =!= _dfTmp("_userId")
      )
      .select(
        df("itemId"),
        df("userId"),
        _dfTmp("_userId"),
        df("score"),
        _dfTmp("_score")
      )
    _df.show(false)

    //  两两向量的维度乘积并加总
    val df_mol = _df.select(
      col("userId"),
      col("_userId"),
      (col("score") * col("_score"))
        .as("score_mol") //用户a和用户b对各自对同一个物品打分的乘积
    ).groupBy(col("userId"),col("_userId"))
      .agg(sum("score_mol"))  //加总
      .withColumnRenamed(
        "sum(score_mol)",
        "mol"
      )
    df_mol.show(false)

    // 计算两两向量的余弦相似度
    val _dfScoreMod = df_score_mod.select(
      col("userId").as("_userId"),
      col("mod").as("_mod")
    )
    //分子表(df_mol)和分母表(df_score_mod)都放在一张表里
    val sim =  df_mol.join(
      df_score_mod,
      df_mol("userId") === df_score_mod("userId")
    ).join(
      _dfScoreMod,
      df_mol("_userId") === _dfScoreMod("_userId")
    ).select(
      df_mol("userId"),
      df_mol("_userId"),
      df_mol("mol"),
      df_score_mod("mod"),
      _dfScoreMod("_mod")
    )
    sim.show(false)

    // 分子 / 分母
    val cos_sim = sim.select(
      col("userId"),
      col("_userId"),
      (col("mol") /
        (col("mod") * col("_mod")))
        .as("cos_sim")
    )
    cos_sim.show(false)
    // 列出某个用户的TopN相似用户

    val topN = cos_sim.rdd.map(x=>(
      (x(0).toString,
        (x(1).toString,x(2).toString)
      )  // 形成 (uid1,(uid2,cos_sim))
      )).groupByKey()
      .mapValues(_.toArray.sortWith((x,y)=>x._2 > y._2)) //根据相似度排序
      .flatMapValues(x=>x)
      .toDF("userId","sim_sort")
      .select(
        col("userId"),
        col("sim_sort._1").as("_userId"),
        col("sim_sort._2").as("cos_sim")
      ).where(col("userId") === "1")

    topN.show(false)



  }


}
