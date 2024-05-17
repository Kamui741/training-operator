# Choose a base image that includes a shell
FROM alpine:latest


RUN apk add --no-cache bash openssl curl kubectl
# Copy the script into the image
COPY gencerts.sh /usr/bin/gencerts.sh

# Make the script executable
RUN chmod +x /usr/bin/gencerts.sh

# Set the entrypoint to your script
ENTRYPOINT ["/usr/bin/gencerts.sh"]
