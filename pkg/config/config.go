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
)

// Config is the global configuration for the training operator.
var Config struct {
	PyTorchInitContainerTemplateFile string
	PyTorchInitContainerImage        string
	MPIKubectlDeliveryImage          string
	PyTorchInitContainerMaxTries     int
}

const (
	// PyTorchInitContainerImageDefault is the default image for the pytorch
	// init container.
	PyTorchInitContainerImageDefault = "alpine:3.10"
	// PyTorchInitContainerTemplateFileDefault is the default template file for
	// the pytorch init container.
	PyTorchInitContainerTemplateFileDefault = "/etc/config/initContainer.yaml"
	// PyTorchInitContainerMaxTriesDefault is the default number of tries for the pytorch init container.
	PyTorchInitContainerMaxTriesDefault = 100
	// MPIKubectlDeliveryImageDefault is the default image for launcher pod in MPIJob init container.
	MPIKubectlDeliveryImageEnv = "MPI_KUBECTL_DELIVERY_IMAGE"
    MPIKubectlDeliveryImageDefault = "kubeflow/kubectl-delivery:latest"
)

//新增读取环境变量的功能
func init() {
	// 初始化读取环境变量的功能
	Config.MPIKubectlDeliveryImage = getEnv(MPIKubectlDeliveryImageEnv, MPIKubectlDeliveryImageDefault)
}


func getEnv(key, defaultValue string) string {
    if value,exists := os.LookupEnv(key); exists {
        return value
    }
	return defaultValue
}