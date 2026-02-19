# Terraform: Deploy Agent Memory DB server via Helm
#
# Prerequisites: kubectl configured, cluster running
# Usage:
#   terraform init
#   terraform plan -var="api_key_secret=your-secret"
#   terraform apply

terraform {
  required_version = ">= 1.0"
  required_providers {
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.0"
    }
  }
}

provider "helm" {
  kubernetes {
    config_path = var.kube_config_path
  }
}

provider "kubernetes" {
  config_path = var.kube_config_path
}

variable "kube_config_path" {
  description = "Path to kubeconfig (default: ~/.kube/config)"
  type        = string
  default     = null
}

variable "namespace" {
  description = "Kubernetes namespace"
  type        = string
  default     = "default"
}

variable "release_name" {
  description = "Helm release name"
  type        = string
  default     = "agent-mem-server"
}

variable "image_repository" {
  description = "Docker image repository"
  type        = string
  default     = "agent-mem-server"
}

variable "image_tag" {
  description = "Docker image tag"
  type        = string
  default     = "latest"
}

variable "api_key_secret" {
  description = "API key for authentication (create K8s secret manually if empty)"
  type        = string
  default     = ""
  sensitive   = true
}

variable "replica_count" {
  description = "Number of replicas"
  type        = number
  default     = 1
}

resource "kubernetes_namespace" "agent_mem" {
  count = var.namespace == "default" ? 0 : 1

  metadata {
    name = var.namespace
  }
}

resource "kubernetes_secret" "api_key" {
  count = var.api_key_secret != "" ? 1 : 0

  metadata {
    name      = "agent-mem-api-key"
    namespace = var.namespace
  }

  data = {
    "api-key" = var.api_key_secret
  }
}

resource "helm_release" "agent_mem_server" {
  name  = var.release_name
  chart = "${path.module}/../helm/agent-mem-server"

  namespace = var.namespace

  values = [
    yamlencode({
      replicaCount = var.replica_count
      image = {
        repository = var.image_repository
        tag       = var.image_tag
        pullPolicy = "IfNotPresent"
      }
      apiKey = var.api_key_secret != "" ? {
        secretName = kubernetes_secret.api_key[0].metadata[0].name
        secretKey  = "api-key"
      } : { secretName = "", secretKey = "api-key" }
      service = {
        type = "ClusterIP"
        port = 8080
      }
    })
  ]

  depends_on = [kubernetes_secret.api_key]
}

output "service_name" {
  value       = "${var.release_name}-agent-mem-server"
  description = "Kubernetes service name for port-forward or ingress"
}
