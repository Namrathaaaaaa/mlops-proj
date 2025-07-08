output "instance_public_ip" {
  description = "The public IP address of the Streamlit EC2 instance."
  value       = aws_instance.streamlit.public_ip
}
