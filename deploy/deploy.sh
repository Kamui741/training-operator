###
 # @Author: ChZheng
 # @Date: 2024-05-14 19:33:19
 # @LastEditTime: 2024-05-17 11:43:56
 # @LastEditors: ChZheng
 # @Description:
 # @FilePath: /training-operator/deploy/deploy.sh
###
echo "开始部署..."
kubectl create -f namespace.yaml
kubectl create -f rbac.yaml
kubectl create -f init.yaml
kubectl create -f kubeflow.org_tfjobs.yaml
kubectl create -f kubeflow.org_mxjobs.yaml
kubectl create -f kubeflow.org_pytorchjobs.yaml
kubectl create -f kubeflow.org_xgboostjobs.yaml
kubectl create -f kubeflow.org_mpijobs.yaml
kubectl create -f kubeflow.org_paddlejobs.yaml
kubectl create -f webhook.yaml
# ./gencerts.sh -n kubeflow
kubectl create -f deployment.yaml
echo "部署完成。"