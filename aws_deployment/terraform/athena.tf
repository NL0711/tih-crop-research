# Glue Data Catalog Database for Crop Analytics
resource "aws_glue_catalog_database" "crop_db" {
  name        = "${replace(var.project_name, "-", "_")}_analytics_db"
  description = "Glue database containing crop disease and prediction analytics tables"
}

# Glue Data Catalog Table representing model prediction logs in S3
resource "aws_glue_catalog_table" "predictions_table" {
  name          = "prediction_logs"
  database_name = aws_glue_catalog_database.crop_db.name
  table_type    = "EXTERNAL_TABLE"

  parameters = {
    "classificationLib" = "org.openx.data.jsonserde.JsonSerDe"
    "serde.serialization.lib" = "org.openx.data.jsonserde.JsonSerDe"
  }

  serialization_provisioned_inputs = []

  storage_descriptor {
    location      = "s3://${aws_s3_bucket.logs.bucket}/prediction-logs/"
    input_format  = "org.apache.hadoop.mapred.TextInputFormat"
    output_format = "org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat"

    ser_de_info {
      name                  = "json"
      serialization_library = "org.openx.data.jsonserde.JsonSerDe"
      parameters = {
        "paths" = "timestamp,class_name,confidence,latency_ms,client_ip"
      }
    }

    columns {
      name = "timestamp"
      type = "string"
    }

    columns {
      name = "class_name"
      type = "string"
    }

    columns {
      name = "confidence"
      type = "double"
    }

    columns {
      name = "latency_ms"
      type = "double"
    }

    columns {
      name = "client_ip"
      type = "string"
    }
  }
}

# Athena Workgroup
resource "aws_athena_workgroup" "crop_workgroup" {
  name = "${var.project_name}-athena-workgroup"

  configuration {
    enforce_workgroup_configuration    = true
    publish_cloudwatch_metrics_enabled = true

    result_configuration {
      output_location = "s3://${aws_s3_bucket.logs.bucket}/athena-query-results/"
      encryption_configuration {
        encryption_option = "SSE_S3"
      }
    }
  }

  tags = local.common_tags
}
