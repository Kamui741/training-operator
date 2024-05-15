###
 # @Author: ChZheng
 # @Date: 2024-05-14 19:33:19
 # @LastEditTime: 2024-05-14 19:33:21
 # @LastEditors: ChZheng
 # @Description:
 # @FilePath: /training-operator/deploy/deploy.sh
###
echo "开始部署..."
kubectl apply -f namespace.yaml
kubectl apply -f rbac.yaml
kubectl apply -f kubeflow.org_mpijobs.yaml
kubectl apply -f webhook.yaml
kubectl apply -f deployment.yaml
echo "部署完成。"