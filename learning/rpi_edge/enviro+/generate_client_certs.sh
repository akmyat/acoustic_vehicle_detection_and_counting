openssl genrsa -out client.key 2048
 openssl req -new -out client.csr -key client.key
 openssl x509 -req -in client.csr -CA ca.crt -CAkey my-ca.key -CAcreateserial -out client.crt -days 365