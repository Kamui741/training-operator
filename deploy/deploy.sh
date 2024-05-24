###
 # @Author: ChZheng
 # @Date: 2024-05-14 19:33:19
 # @LastEditTime: 2024-05-23 09:21:10
 # @LastEditors: ChZheng
 # @Description:
 # @FilePath: /training-operator/deploy/deploy.sh
###
echo "开始部署..."
kubectl create -f namespace.yaml
kubectl create -f service-account.yaml #顺序
kubectl create -f role.yaml
kubectl create -f cluster-role-binding.yaml
kubectl create -f init.yaml
kubectl create -f crd_tfjobs.yaml
kubectl create -f crd_mxjobs.yaml
kubectl create -f crd_pytorchjobs.yaml
kubectl create -f crd_xgboostjobs.yaml
kubectl create -f crd_mpijobs.yaml
kubectl create -f crd_paddlejobs.yaml
kubectl create -f webhook.yaml
kubectl create -f service.yaml
kubectl create -f deployment.yaml
echo "部署完成。"