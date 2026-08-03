variable "aws_region" {
  type        = string
  default     = "us-east-1"
  description = "AWS region to deploy resources"
}

variable "project_name" {
  type        = string
  default     = "tih-crop-research"
  description = "Project name used for resource naming prefix"
}

variable "environment" {
  type        = string
  default     = "prod"
  description = "Target deployment environment (e.g. dev, staging, prod)"
}

variable "app_port" {
  type        = number
  default     = 8000
  description = "Port exposed by the FastAPI container"
}

variable "image_tag" {
  type        = string
  default     = "latest"
  description = "Docker image tag to deploy to AWS App Runner"
}

variable "app_runner_cpu" {
  type        = string
  default     = "1024" # 1 vCPU
  description = "CPU configuration for App Runner (1024 = 1 vCPU, 2048 = 2 vCPU)"
}

variable "app_runner_memory" {
  type        = string
  default     = "2048" # 2 GB
  description = "Memory configuration for App Runner (2048 = 2 GB, 4096 = 4 GB)"
}
