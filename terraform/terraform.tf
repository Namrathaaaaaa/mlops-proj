terraform {
  required_providers {
    aws = {
      source = "hashicorp/aws"
      version = "6.0.0-beta2"
    }
  }
  #   backend "s3" {
  #   bucket         = "mlops-terraform-state-bucket-remote"
  #   key            = "mlops/terraform.tfstate"
  #   region         = "us-east-1"
  #   use_lockfile  = true
  #   encrypt        = true
  # }
}


   