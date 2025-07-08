variable "ami" {
  description = "AMI ID for the EC2 instance."
  type        = string
  default     = ""############## put ur ami id
}

variable "instance_type" {
  description = "EC2 instance type."
  type        = string
  default     = "t2.micro"
}

variable "docker_image" {
  description = "Docker image for the Streamlit app."
  type        = string
  default     = ""############## put ur docker image
}

variable "key_name" {
  description = "Key name for the EC2 instance."
  type        = string
  default     = "aws-key"
}
