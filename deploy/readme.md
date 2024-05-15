<!--
 * @Author: ChZheng
 * @Date: 2024-05-14 16:30:44
 * @LastEditTime: 2024-05-14 17:21:05
 * @LastEditors: ChZheng
 * @Description:
 * @FilePath: /training-operator/deploy/readme.md
-->
部署顺序
namespace.yaml
rbac.yaml
kubeflow.org_mpijobs.yaml
webhook.yaml
deployment.yaml

namespace/kubeflow created

customresourcedefinition.apiextensions.k8s.io/mpijobs.kubeflow.org created

kubeflow.org created

serviceaccount/training-operator created

clusterrole.rbac.authorization.k8s.io/training-operator created

clusterrolebinding.rbac.authorization.k8s.io/training-operator created

secret/training-operator-webhook-cert created

service/training-operator created

deployment.apps/training-operator created

validatingwebhookconfiguration.admissionregistration.k8s.io/validator.training-operator.kubeflow.org created