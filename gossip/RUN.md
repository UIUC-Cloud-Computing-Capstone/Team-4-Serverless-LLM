docker build -f gossip/src/coordinator/Dockerfile -t gossip-central-coordinator .

docker run -it --rm -p 9000:9000 gossip-central-coordinator
