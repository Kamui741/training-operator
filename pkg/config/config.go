/*
 * @Author: ChZheng
 * @Date: 2024-05-13 19:44:41
 * @LastEditTime: 2024-08-29 19:06:18
 * @LastEditors: ChZheng
 * @Description:
 * @FilePath: /笔记/Users/apple/go/src/github.com/training-operator/pkg/config/config.go
 */
// Copyright 2021 The Kubeflow Authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License

package config

import (
	"os"
	"strconv"
)

// Config is the global configuration for the training operator.
var Config struct {
	PyTorchInitContainerTemplateFile string
	PyTorchInitContainerImage        string
	MPIKubectlDeliveryImage          string
	PyTorchInitContainerMaxTries     int
}

const (
	// Environment variables and their default values
	PyTorchInitContainerTemplateFileEnv     = "PYTORCH_INIT_CONTAINER_TEMPLATE_FILE"
	PyTorchInitContainerTemplateFileDefault = "/etc/config/initContainer.yaml"
	PyTorchInitContainerMaxTriesEnv         = "PYTORCH_INIT_CONTAINER_MAX_TRIES"
	PyTorchInitContainerMaxTriesDefault     = 100
	MPIKubectlDeliveryImageEnv              = "MPI_KUBECTL_DELIVERY_IMAGE"
	MPIKubectlDeliveryImageDefault          = "kubeflow/kubectl-delivery:latest"
	PyTorchInitContainerImageEnv            = "PYTORCH_INIT_CONTAINER_IMAGE"
	PyTorchInitContainerImageDefault        = "alpine:3.10"
)

// Initialize and load configurations from environment variables
func init() {
	Config.PyTorchInitContainerTemplateFile = getEnv(PyTorchInitContainerTemplateFileEnv, PyTorchInitContainerTemplateFileDefault)
	Config.PyTorchInitContainerImage = getEnv(PyTorchInitContainerImageEnv, PyTorchInitContainerImageDefault)
	Config.MPIKubectlDeliveryImage = getEnv(MPIKubectlDeliveryImageEnv, MPIKubectlDeliveryImageDefault)
	Config.PyTorchInitContainerMaxTries = getEnvAsInt(PyTorchInitContainerMaxTriesEnv, PyTorchInitContainerMaxTriesDefault)
}

// getEnv retrieves the value of the environment variable named by the key,
// or returns the provided default value if the variable is not present.
func getEnv(key, defaultValue string) string {
	if value, exists := os.LookupEnv(key); exists {
		return value
	}
	return defaultValue
}

// getEnvAsInt retrieves the value of the environment variable as an integer,
// or returns the provided default value if the variable is not present or is not a valid integer.
func getEnvAsInt(key string, defaultValue int) int {
	if value, exists := os.LookupEnv(key); exists {
		if intValue, err := strconv.Atoi(value); err == nil {
			return intValue
		}
	}
	return defaultValue
}
