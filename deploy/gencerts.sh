#!/bin/bash


set -e
SCRIPT=`basename ${BASH_SOURCE[0]}`
# RESOURCE_NAME="spark-webhook-certs" # 不再需要，因为我们将使用新的资源名称

function usage {
  cat<< EOF
  Usage: $SCRIPT
  Options:
  -h | --help                    Display help information.
  -n | --namespace <namespace>   The namespace where the Training operator is installed.
  -s | --service <service>       The name of the webhook service.
  -p | --in-pod                  Whether the script is running inside a pod or not.
  -r | --resource-name           The training resource name that will hold the secret [default: $RESOURCE_NAME]
EOF
}

function parse_arguments {
  # 默认的资源名称变量更新为适用于trainingoperator
  RESOURCE_NAME="training-operator-webhook-cert" # 新的默认资源名称
  while [[ $# -gt 0 ]]
  do
    case "$1" in
      -n|--namespace)
      if [[ -n "$2" ]]; then
        NAMESPACE="$2"
      else
        echo "-n or --namespace requires a value."
        exit 1
      fi
      shift 2
      continue
      ;;
      -s|--service)
      if [[ -n "$2" ]]; then
        SERVICE="$2"
      else
        echo "-s or --service requires a value."
        exit 1
      fi
      shift 2
      continue
      ;;
      -p|--in-pod)
      export IN_POD=true
      shift 1
      continue
      ;;
      -r|--resource-name)
      if [[ -n "$2" ]]; then
        RESOURCE_NAME="$2"
      else
        echo "-r or --resource-name requires a value."
        exit 1
      fi
      shift 2
      continue
      ;;
      -h|--help)
      usage
      exit 0
      ;;
      --)              # End of all options.
        shift
        break
      ;;
      '')              # End of all options.
        break
      ;;
      *)
        echo "Unrecognized option: $1"
        exit 1
      ;;
    esac
  done
}

# Set the namespace to "training-operator" by default if not provided.
# Set the webhook service name to "training-webhook" by default if not provided.
IN_POD=false
SERVICE="" # 更新服务名称
NAMESPACE="hxz-cabt" # 更新命名空间
parse_arguments "$@"

TMP_DIR="/tmp/training-pod-webhook-certs" # 更新临时目录名称

echo "Generating certs for the Training pod admission webhook in ${TMP_DIR}."
mkdir -p ${TMP_DIR}
cat > ${TMP_DIR}/server.conf << EOF
[req]
req_extensions = v3_req
distinguished_name = req_distinguished_name
[req_distinguished_name]
[ v3_req ]
basicConstraints = CA:FALSE
keyUsage = nonRepudiation, digitalSignature, keyEncipherment
extendedKeyUsage = clientAuth, serverAuth
subjectAltName = DNS:${SERVICE}.${NAMESPACE}.svc
EOF

# Create a certificate authority.
touch ${TMP_DIR}/.rnd
export RANDFILE=${TMP_DIR}/.rnd
openssl genrsa -out ${TMP_DIR}/ca-key.pem 2048
openssl req -x509 -new -nodes -key ${TMP_DIR}/ca-key.pem -days 100000 -out ${TMP_DIR}/ca-cert.pem -subj "/CN=${SERVICE}.${NAMESPACE}.svc"

# Create a server certificate.
openssl genrsa -out ${TMP_DIR}/server-key.pem 2048
# Note the CN is the DNS name of the service of the webhook.
openssl req -new -key ${TMP_DIR}/server-key.pem -out ${TMP_DIR}/server.csr -subj "/CN=${SERVICE}.${NAMESPACE}.svc" -config ${TMP_DIR}/server.conf
openssl x509 -req -in ${TMP_DIR}/server.csr -CA ${TMP_DIR}/ca-cert.pem -CAkey ${TMP_DIR}/ca-key.pem -CAcreateserial -out ${TMP_DIR}/server-cert.pem -days 100000 -extensions v3_req -extfile ${TMP_DIR}/server.conf

if [[ "$IN_POD" == "true" ]];  then
  TOKEN=$(cat /var/run/secrets/kubernetes.io/serviceaccount/token)

  # Base64 encode secrets and then remove the trailing newline to avoid issues in the curl command
  ca_cert=$(cat ${TMP_DIR}/ca-cert.pem | base64 | tr -d '\n')
  ca_key=$(cat ${TMP_DIR}/ca-key.pem | base64 | tr -d '\n')
  server_cert=$(cat ${TMP_DIR}/server-cert.pem | base64 | tr -d '\n')
  server_key=$(cat ${TMP_DIR}/server-key.pem | base64 | tr -d '\n')

  # Create the secret resource
  echo "Creating a secret for the certificate and keys"
  STATUS=$(curl -ik \
	  -o ${TMP_DIR}/output \
	  -w "%{http_code}" \
	  -X POST \
	  -H "Authorization: Bearer $TOKEN" \
	  -H 'Accept: application/json' \
	  -H 'Content-Type: application/json' \
	  -d '{
	  "kind": "Secret",
	  "apiVersion": "v1",
	  "metadata": {
	    "name": "'"$RESOURCE_NAME"'",
	    "namespace": "'"$NAMESPACE"'"
	  },
	  "data": {
	    "ca-cert.pem": "'"$ca_cert"'",
	    "ca-key.pem": "'"$ca_key"'",
	    "server-cert.pem": "'"$server_cert"'",
	    "server-key.pem": "'"$server_key"'"
	  }
	}' \
	https://kubernetes.default.svc/api/v1/namespaces/${NAMESPACE}/secrets)

  cat ${TMP_DIR}/output

  case "$STATUS" in
    201)
      printf "\nSuccess - secret created.\n"
    ;;
    409)
      printf "\nSuccess - secret already exists.\n"
     ;;
     *)
      printf "\nFailed creating secret.\n"
      exit 1
     ;;
  esac
else
  # 使用kubectl命令创建Secret资源，命名空间和资源名称已更新
  kubectl create secret --namespace=${NAMESPACE} generic ${RESOURCE_NAME} --from-file=${TMP_DIR}/ca-key.pem --from-file=${TMP_DIR}/ca-cert.pem --from-file=${TMP_DIR}/server-key.pem --from-file=${TMP_DIR}/server-cert.pem
fi

# Clean up after we're done.
printf "\nDeleting ${TMP_DIR}.\n"
rm -rf ${TMP_DIR}
