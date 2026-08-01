resource "aws_apprunner_service" "api" {
  service_name = "${var.project_name}-service"

  source_configuration {
    authentication_configuration {
      access_role_arn = aws_iam_role.apprunner_access.arn
    }
    image_repository {
      image_identifier      = "${aws_ecr_repository.api.repository_url}:${var.image_tag}"
      image_repository_type = "ECR"
      
      image_configuration {
        port = var.app_port
        
        runtime_environment_variables = {
          MODEL_S3_BUCKET  = "${var.project_name}-artifacts-${var.environment}"
          MODEL_S3_KEY     = "best_ckpt.pth"
          MODEL_CONFIG     = "/app/classification/configs/DAMamba/damamba_tiny.yaml"
          MODEL_CHECKPOINT = "/app/output/tiny/finetune/best_ckpt.pth"
          PYTHONPATH       = "/app"
        }
      }
    }
    auto_deployments_enabled = false
  }

  instance_configuration {
    cpu               = var.app_runner_cpu
    memory            = var.app_runner_memory
    instance_role_arn = aws_iam_role.apprunner_instance.arn
  }

  tags = local.common_tags
}
