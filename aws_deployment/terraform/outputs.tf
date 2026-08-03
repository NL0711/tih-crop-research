output "ecr_repository_url" {
  value       = aws_ecr_repository.api.repository_url
  description = "The URL of the ECR repository"
}

output "app_runner_url" {
  value       = aws_apprunner_service.api.service_url
  description = "The public endpoint URL of the FastAPI crop inference service"
}

output "s3_bucket_datasets" {
  value       = aws_s3_bucket.datasets.id
  description = "Name of the S3 bucket for training datasets"
}

output "s3_bucket_artifacts" {
  value       = aws_s3_bucket.artifacts.id
  description = "Name of the S3 bucket for model checkpoints and weights"
}

output "sagemaker_role_arn" {
  value       = aws_iam_role.sagemaker_execution.arn
  description = "IAM Role ARN to use for SageMaker training jobs"
}

output "athena_database" {
  value       = aws_glue_catalog_database.crop_db.name
  description = "The name of the Athena/Glue database"
}
