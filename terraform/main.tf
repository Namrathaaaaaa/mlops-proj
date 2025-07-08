resource "aws_key_pair" "deployer" {
  key_name   = "deployer-key"
  public_key = file("keys/fruits-classifier.pub")
}

resource "aws_security_group" "streamlit_sg" {
  name        = "streamlit-sg"
  description = "Allow 8501 and SSH"
  ingress {
    from_port   = 8501
    to_port     = 8501
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

resource "aws_instance" "streamlit" {
  ami           = var.ami
  instance_type = var.instance_type
  key_name      = aws_key_pair.deployer.key_name
  security_groups = [aws_security_group.streamlit_sg.name]

 user_data = <<-EOF
              #!/bin/bash
              exec > /var/log/user-data.log 2>&1
              set -x
              yum update -y
              yum install -y docker
              systemctl start docker
              systemctl enable docker
              usermod -aG docker ec2-user
              docker run -d -p 8501:8501 ${var.docker_image}
              EOF

  tags = {
    Name = "streamlit-mlops"
  }
}