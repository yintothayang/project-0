#!/bin/bash
tar -czvf project-0.tar.gz project-0/
scp -i ~/.ssh/tf.pem project-0.tar.gz ubuntu@ec2-54-187-15-156.us-west-2.compute.amazonaws.com:~/
# tar -xzvf project-0.tar.gz
